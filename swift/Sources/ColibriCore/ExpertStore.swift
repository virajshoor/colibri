// Routed experts streamed from SSD into a RAM cache with a byte budget.
//
// This is what lets a model far larger than memory run: the dense part (attention,
// embeddings, shared experts) stays resident, and of the hundreds of experts per layer
// only the handful the router picks for a token are needed. Each expert is three
// matrices (gate, up, down). A miss reads them with pread straight from the shard
// files; all misses of one layer are read in parallel so the SSD sees a deep queue,
// which is what NVMe drives need to reach their rated bandwidth.
//
// Eviction is least-recently-used, except for pinned experts: at start-up the most
// used experts from the `.coli_usage` history are loaded and never evicted.

import Foundation

/// The three matrices of one routed expert.
public final class ExpertWeights: @unchecked Sendable {
    public let gate: QuantTensor  // [moeInter, hidden]
    public let up: QuantTensor    // [moeInter, hidden]
    public let down: QuantTensor  // [hidden, moeInter]
    public var bytes: Int { gate.residentBytes + up.residentBytes + down.residentBytes }

    public init(gate: QuantTensor, up: QuantTensor, down: QuantTensor) {
        self.gate = gate
        self.up = up
        self.down = down
    }
}

/// Counters reported by `coli` and the server.
public struct ExpertStats: Sendable {
    public var hits = 0
    public var misses = 0
    public var bytesRead: Int64 = 0
    public var readSeconds = 0.0
    public var cachedExperts = 0
    public var cachedBytes: Int64 = 0
    public var pinnedExperts = 0
    public var hitRate: Double { hits + misses == 0 ? 0 : Double(hits) / Double(hits + misses) }
    public init() {}
}

public final class ExpertStore: @unchecked Sendable {
    let shards: ShardSet
    let config: ModelConfig
    /// Maximum bytes of cached experts. Pinned experts count against it.
    public let budget: Int64

    private struct Entry {
        let weights: ExpertWeights
        var lastUse: UInt64
        var pinned: Bool
    }

    private let lock = NSLock()
    private var cache: [Int: Entry] = [:]
    private var clock: UInt64 = 0
    private var cachedBytes: Int64 = 0
    private var stats = ExpertStats()
    /// Selections per (layer, expert), persisted to `.coli_usage`.
    public private(set) var usage: [[UInt32]]

    public init(shards: ShardSet, config: ModelConfig, budget: Int64) {
        self.shards = shards
        self.config = config
        self.budget = max(0, budget)
        usage = Array(repeating: Array(repeating: 0, count: config.experts), count: config.layers)
    }

    @inline(__always) private func key(_ layer: Int, _ id: Int) -> Int { layer * config.experts + id }

    /// Tensor name of one expert matrix, as written by the converter.
    public static func tensorName(layer: Int, expert: Int, part: String) -> String {
        "model.layers.\(layer).mlp.experts.\(expert).\(part).weight"
    }

    /// Reads one expert from disk. Called without the lock held.
    public func loadFromDisk(layer: Int, expert: Int) throws -> ExpertWeights {
        let c = config
        let g = try QuantTensor.load(shards, Self.tensorName(layer: layer, expert: expert, part: "gate_proj"),
                                     rows: c.moeInter, cols: c.hidden)
        let u = try QuantTensor.load(shards, Self.tensorName(layer: layer, expert: expert, part: "up_proj"),
                                     rows: c.moeInter, cols: c.hidden)
        let d = try QuantTensor.load(shards, Self.tensorName(layer: layer, expert: expert, part: "down_proj"),
                                     rows: c.hidden, cols: c.moeInter)
        return ExpertWeights(gate: g, up: u, down: d)
    }

    /// Returns the weights of every expert in `ids` for `layer`, reading the missing
    /// ones from disk in parallel, and counts the selection in the usage history.
    public func fetch(layer: Int, ids: [Int]) throws -> [Int: ExpertWeights] {
        var found: [Int: ExpertWeights] = [:]
        var missing: [Int] = []
        lock.lock()
        for id in ids where found[id] == nil && !missing.contains(id) {
            usage[layer][id] &+= 1
            clock += 1
            if var e = cache[key(layer, id)] {
                e.lastUse = clock
                cache[key(layer, id)] = e
                found[id] = e.weights
                stats.hits += 1
            } else {
                missing.append(id)
                stats.misses += 1
            }
        }
        lock.unlock()
        if missing.isEmpty { return found }

        // Parallel reads: one task per missing expert.
        let t0 = monotonicSeconds()
        var loaded = [ExpertWeights?](repeating: nil, count: missing.count)
        var failure: Error?
        let resultLock = NSLock()
        DispatchQueue.concurrentPerform(iterations: missing.count) { k in
            do {
                let w = try self.loadFromDisk(layer: layer, expert: missing[k])
                resultLock.lock(); loaded[k] = w; resultLock.unlock()
            } catch {
                resultLock.lock(); failure = error; resultLock.unlock()
            }
        }
        if let failure { throw failure }
        let dt = monotonicSeconds() - t0

        lock.lock()
        stats.readSeconds += dt
        for (k, id) in missing.enumerated() {
            let w = loaded[k]!
            found[id] = w
            stats.bytesRead += Int64(w.bytes)
            insertLocked(layer: layer, id: id, weights: w, pinned: false)
        }
        lock.unlock()
        return found
    }

    /// Adds an expert to the cache and evicts least-recently-used unpinned experts
    /// until the cache fits its budget. The caller holds the lock.
    private func insertLocked(layer: Int, id: Int, weights: ExpertWeights, pinned: Bool) {
        guard budget > 0 else { return }  // budget 0: stream every expert, cache nothing
        let k = key(layer, id)
        if cache[k] != nil { return }
        clock += 1
        cache[k] = Entry(weights: weights, lastUse: clock, pinned: pinned)
        cachedBytes += Int64(weights.bytes)
        if pinned { stats.pinnedExperts += 1 }
        while cachedBytes > budget {
            var victim: Int?
            var oldest = UInt64.max
            for (kk, e) in cache where !e.pinned && e.lastUse < oldest {
                oldest = e.lastUse
                victim = kk
            }
            guard let v = victim, let e = cache.removeValue(forKey: v) else { break }
            cachedBytes -= Int64(e.weights.bytes)
        }
    }

    /// Loads the given experts and marks them non-evictable. Stops when the pins would
    /// take more than `maxBytes`. Returns how many were pinned.
    @discardableResult
    public func pin(_ list: [(layer: Int, expert: Int)], maxBytes: Int64) throws -> Int {
        var used: Int64 = 0
        var n = 0
        // Read in parallel batches of 32 so start-up uses the whole SSD queue.
        var i = 0
        while i < list.count {
            let batch = Array(list[i..<min(list.count, i + 32)])
            i += batch.count
            var loaded = [ExpertWeights?](repeating: nil, count: batch.count)
            let l = NSLock()
            DispatchQueue.concurrentPerform(iterations: batch.count) { k in
                let w = try? self.loadFromDisk(layer: batch[k].layer, expert: batch[k].expert)
                l.lock(); loaded[k] = w; l.unlock()
            }
            lock.lock()
            for (k, item) in batch.enumerated() {
                guard let w = loaded[k] else { continue }
                if used + Int64(w.bytes) > maxBytes { lock.unlock(); return n }
                used += Int64(w.bytes)
                insertLocked(layer: item.layer, id: item.expert, weights: w, pinned: true)
                n += 1
            }
            lock.unlock()
        }
        return n
    }

    public func isCached(layer: Int, expert: Int) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return cache[key(layer, expert)] != nil
    }

    public func snapshot() -> ExpertStats {
        lock.lock(); defer { lock.unlock() }
        var s = stats
        s.cachedExperts = cache.count
        s.cachedBytes = cachedBytes
        return s
    }

    /// Replaces the usage counters (when a `.coli_usage` history is loaded).
    public func setUsage(_ u: [[UInt32]]) {
        lock.lock(); defer { lock.unlock() }
        if u.count == usage.count { usage = u }
    }

    public func usageSnapshot() -> [[UInt32]] {
        lock.lock(); defer { lock.unlock() }
        return usage
    }

    /// Bytes of one expert as stored on disk (weights + scales), from the first sparse
    /// layer. Used to plan the cache before anything is loaded.
    public func bytesPerExpert() -> Int64 {
        let l = config.firstDense
        var total: Int64 = 0
        for part in ["gate_proj", "up_proj", "down_proj"] {
            let n = Self.tensorName(layer: l, expert: 0, part: part)
            total += Int64(shards.tensors[n]?.byteCount ?? 0) + Int64(shards.tensors[n + ".qs"]?.byteCount ?? 0)
        }
        return total
    }
}
