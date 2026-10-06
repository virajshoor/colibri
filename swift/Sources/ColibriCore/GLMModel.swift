// The GLM-5 / DeepSeek-V3 decoder: multi-head latent attention (MLA) and a
// mixture-of-experts feed-forward whose routed experts stream from SSD.
//
// One layer is
//     x += Attention(RMSNorm(x))
//     x += FFN(RMSNorm(x))         FFN = dense MLP for the first layers, MoE after
//
// MLA keeps a compressed KV cache: per token and layer only a normalised latent
// (kv_lora values) and one shared rotary key (qk_rope values) are stored, not full
// per-head keys and values. Attention never expands the latent; instead each head's
// query is folded through kv_b into latent space (the "absorbed" form):
//     score(t) = (W_kᵀ q_nope) · latent_t + q_rope · k_rope_t
//     out      = W_v (Σ_t p_t latent_t)
// which is the same arithmetic with far less work per cached token.
//
// GLM-5's DSA indexer only restricts attention once the context exceeds `index_topk`
// tokens; below that it selects every key, so full attention here is exact. Beyond it
// this engine still attends to every token (a superset of what DSA would pick).

import Foundation

/// Attention projections of one layer.
final class AttentionWeights {
    let qA: QuantTensor?       // [qLora, D]           (nil when q_lora_rank == 0)
    let qANorm: [Float]        // [qLora]
    let qB: QuantTensor        // [H*qkHead, qLora] or q_proj [H*qkHead, D]
    let kvA: QuantTensor       // [kvLora + qkRope, D]
    let kvANorm: [Float]       // [kvLora]
    let kvB: QuantTensor       // [H*(qkNope+vHead), kvLora]
    let o: QuantTensor         // [D, H*vHead]

    init(qA: QuantTensor?, qANorm: [Float], qB: QuantTensor, kvA: QuantTensor, kvANorm: [Float],
         kvB: QuantTensor, o: QuantTensor) {
        self.qA = qA; self.qANorm = qANorm; self.qB = qB
        self.kvA = kvA; self.kvANorm = kvANorm; self.kvB = kvB; self.o = o
    }

    var bytes: Int {
        (qA?.residentBytes ?? 0) + qB.residentBytes + kvA.residentBytes + kvB.residentBytes
            + o.residentBytes + 4 * (qANorm.count + kvANorm.count)
    }
}

/// gate/up/down of a dense MLP or of the shared expert.
final class MLPWeights {
    let gate: QuantTensor, up: QuantTensor, down: QuantTensor
    init(gate: QuantTensor, up: QuantTensor, down: QuantTensor) { self.gate = gate; self.up = up; self.down = down }
    var bytes: Int { gate.residentBytes + up.residentBytes + down.residentBytes }

    /// out[S, D] = down(silu(gate(x)) * up(x)).
    func forward(_ x: ConstFloatPtr, count s: Int, into out: FloatPtr) {
        let inter = gate.rows
        let g = FloatPtr.allocate(capacity: s * inter), u = FloatPtr.allocate(capacity: s * inter)
        defer { g.deallocate(); u.deallocate() }
        gate.matmul(x, count: s, into: g)
        up.matmul(x, count: s, into: u)
        for i in 0..<(s * inter) { g[i] = silu(g[i]) * u[i] }
        down.matmul(g, count: s, into: out)
    }
}

final class LayerWeights {
    let inputNorm: [Float]
    let postNorm: [Float]
    let attention: AttentionWeights
    let dense: MLPWeights?          // dense layers
    let router: QuantTensor?        // [E, D] f32, sparse layers
    let routerBias: [Float]         // [E] e_score_correction_bias
    let shared: MLPWeights?         // shared expert(s), sparse layers

    init(inputNorm: [Float], postNorm: [Float], attention: AttentionWeights, dense: MLPWeights?,
         router: QuantTensor?, routerBias: [Float], shared: MLPWeights?) {
        self.inputNorm = inputNorm; self.postNorm = postNorm; self.attention = attention
        self.dense = dense; self.router = router; self.routerBias = routerBias; self.shared = shared
    }

    var isSparse: Bool { router != nil }
    var bytes: Int {
        attention.bytes + (dense?.bytes ?? 0) + (router?.residentBytes ?? 0) + (shared?.bytes ?? 0)
            + 4 * (inputNorm.count + postNorm.count + routerBias.count)
    }
}

/// The compressed KV cache of one conversation, plus the tokens it holds so a later
/// request with the same prefix can reuse it.
public final class KVCache {
    var latent: [[Float]]   // per layer: [tokens * kvLora]
    var rope: [[Float]]     // per layer: [tokens * qkRope]
    public private(set) var tokens: [Int] = []
    let kvLora: Int, qkRope: Int

    public init(config: ModelConfig) {
        latent = Array(repeating: [], count: config.layers)
        rope = Array(repeating: [], count: config.layers)
        kvLora = config.kvLora
        qkRope = config.qkRope
    }

    public var count: Int { tokens.count }

    /// Drops everything after the first `n` tokens.
    public func truncate(to n: Int) {
        guard n < tokens.count else { return }
        tokens.removeLast(tokens.count - n)
        for l in 0..<latent.count {
            latent[l].removeLast(latent[l].count - n * kvLora)
            rope[l].removeLast(rope[l].count - n * qkRope)
        }
    }

    func appendTokens(_ t: [Int]) { tokens += t }

    /// Bytes held by the cache.
    public var bytes: Int { latent.reduce(0) { $0 + $1.count * 4 } + rope.reduce(0) { $0 + $1.count * 4 } }
}

/// Timing of the last forward pass.
public struct ForwardTiming: Sendable {
    public var attention = 0.0
    public var experts = 0.0
    public var expertWait = 0.0
    public var head = 0.0
    public init() {}
}

public final class GLMModel: @unchecked Sendable {
    public let config: ModelConfig
    public let directory: URL
    public let shards: ShardSet
    public let experts: ExpertStore
    let embed: QuantTensor        // [vocab, D]
    let lmHead: QuantTensor       // [vocab, D]
    let finalNorm: [Float]
    let layers: [LayerWeights]
    /// Bytes of resident (non-expert) weights.
    public let denseBytes: Int

    /// Experts per token: nil uses the model's num_experts_per_tok; a smaller value
    /// trades quality for less disk traffic.
    public var expertTopK: Int?
    /// Adaptive expert selection: keep the smallest set of top experts whose routing
    /// weight reaches this fraction of the total (0 disables).
    public var expertTopP: Float = 0
    public private(set) var lastTiming = ForwardTiming()
    private var warnedDSA = false

    /// Opens a converted model directory. Dense weights are read into RAM now; experts
    /// are read on demand into a cache of `expertBudget` bytes.
    public init(directory: URL, expertBudget: (_ denseBytes: Int64) -> Int64,
                progress: ((String) -> Void)? = nil) throws {
        // Everything is built in locals first: a class may not read its own stored
        // properties until all of them are set.
        let c = try ModelConfig.load(directory: directory)
        let s = try ShardSet(directory: directory)
        progress?("indexing \(s.tensors.count) tensors in \(s.files.count) shards")

        let embedT = try QuantTensor.load(s, "model.embed_tokens.weight", rows: c.vocab, cols: c.hidden)
        let headT = try s.has("lm_head.weight")
            ? QuantTensor.load(s, "lm_head.weight", rows: c.vocab, cols: c.hidden) : embedT
        let norm = try s.readFloats("model.norm.weight", expectCount: c.hidden)

        // Layers load in parallel: each is an independent set of reads.
        var loaded = [LayerWeights?](repeating: nil, count: c.layers)
        var failure: Error?
        let lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: c.layers) { l in
            do {
                let w = try GLMModel.loadLayer(s, c, l)
                lock.lock(); loaded[l] = w; lock.unlock()
            } catch {
                lock.lock(); failure = error; lock.unlock()
            }
        }
        if let failure { throw failure }
        let layerList = loaded.map { $0! }
        let lmBytes = headT === embedT ? 0 : headT.residentBytes
        let dense = embedT.residentBytes + lmBytes + layerList.reduce(0) { $0 + $1.bytes }
        progress?("resident weights: \(formatBytes(Int64(dense)))")

        self.directory = directory
        config = c
        shards = s
        embed = embedT
        lmHead = headT
        finalNorm = norm
        layers = layerList
        denseBytes = dense
        experts = ExpertStore(shards: s, config: c, budget: expertBudget(Int64(dense)))
    }

    static func loadLayer(_ s: ShardSet, _ c: ModelConfig, _ l: Int) throws -> LayerWeights {
        let p = "model.layers.\(l)."
        func t(_ n: String, _ o: Int, _ i: Int) throws -> QuantTensor { try QuantTensor.load(s, p + n, rows: o, cols: i) }
        func v(_ n: String, _ count: Int) throws -> [Float] { try s.readFloats(p + n, expectCount: count) }
        let h = c.heads
        let qA: QuantTensor?, qANorm: [Float], qB: QuantTensor
        if c.qLora > 0 {
            qA = try t("self_attn.q_a_proj.weight", c.qLora, c.hidden)
            qANorm = try v("self_attn.q_a_layernorm.weight", c.qLora)
            qB = try t("self_attn.q_b_proj.weight", h * c.qkHead, c.qLora)
        } else {
            qA = nil
            qANorm = []
            qB = try t("self_attn.q_proj.weight", h * c.qkHead, c.hidden)
        }
        let attn = AttentionWeights(
            qA: qA, qANorm: qANorm, qB: qB,
            kvA: try t("self_attn.kv_a_proj_with_mqa.weight", c.kvLora + c.qkRope, c.hidden),
            kvANorm: try v("self_attn.kv_a_layernorm.weight", c.kvLora),
            kvB: try t("self_attn.kv_b_proj.weight", h * (c.qkNope + c.vHead), c.kvLora),
            o: try t("self_attn.o_proj.weight", c.hidden, h * c.vHead))
        let inNorm = try v("input_layernorm.weight", c.hidden)
        let postNorm = try v("post_attention_layernorm.weight", c.hidden)
        if l < c.firstDense {
            let mlp = MLPWeights(
                gate: try t("mlp.gate_proj.weight", c.denseInter, c.hidden),
                up: try t("mlp.up_proj.weight", c.denseInter, c.hidden),
                down: try t("mlp.down_proj.weight", c.hidden, c.denseInter))
            return LayerWeights(inputNorm: inNorm, postNorm: postNorm, attention: attn, dense: mlp,
                                router: nil, routerBias: [], shared: nil)
        }
        let router = QuantTensor(rows: c.experts, cols: c.hidden,
                                 values: try v("mlp.gate.weight", c.experts * c.hidden))
        let bias = try s.has(p + "mlp.gate.e_score_correction_bias")
            ? v("mlp.gate.e_score_correction_bias", c.experts) : [Float](repeating: 0, count: c.experts)
        var shared: MLPWeights?
        if c.sharedExperts > 0 {
            let si = c.moeInter * c.sharedExperts
            shared = MLPWeights(
                gate: try t("mlp.shared_experts.gate_proj.weight", si, c.hidden),
                up: try t("mlp.shared_experts.up_proj.weight", si, c.hidden),
                down: try t("mlp.shared_experts.down_proj.weight", c.hidden, si))
        }
        return LayerWeights(inputNorm: inNorm, postNorm: postNorm, attention: attn, dense: nil,
                            router: router, routerBias: bias, shared: shared)
    }

    // MARK: forward pass

    /// Runs `tokens` (appended after what `cache` holds) through the model and returns
    /// the logits for the last one. Prefill passes many tokens; decode passes one.
    public func forward(_ tokens: [Int], cache: KVCache) throws -> [Float] {
        let c = config, d = c.hidden, s = tokens.count
        precondition(s > 0)
        for t in tokens where t < 0 || t >= c.vocab { throw ColibriError("token id \(t) out of range") }
        let pos0 = cache.count
        if c.indexTopK > 0 && pos0 + s > c.indexTopK && !warnedDSA {
            warnedDSA = true
            FileHandle.standardError.write(Data(
                "[colibri] context passed index_topk=\(c.indexTopK): attending to all tokens (DSA selection not applied)\n".utf8))
        }
        lastTiming = ForwardTiming()

        let x = FloatPtr.allocate(capacity: s * d)
        let nrm = FloatPtr.allocate(capacity: s * d)
        let tmp = FloatPtr.allocate(capacity: s * d)
        defer { x.deallocate(); nrm.deallocate(); tmp.deallocate() }
        for (i, t) in tokens.enumerated() { embed.dequantRow(t, into: x + i * d) }

        for (l, layer) in layers.enumerated() {
            let ta = monotonicSeconds()
            layer.inputNorm.withUnsafeBufferPointer { w in
                for i in 0..<s { rmsNorm(nrm + i * d, x + i * d, w.baseAddress!, d, eps: c.eps) }
            }
            attention(layer.attention, layer: l, nrm, count: s, pos0: pos0, cache: cache, into: tmp)
            for i in 0..<(s * d) { x[i] += tmp[i] }
            lastTiming.attention += monotonicSeconds() - ta

            layer.postNorm.withUnsafeBufferPointer { w in
                for i in 0..<s { rmsNorm(nrm + i * d, x + i * d, w.baseAddress!, d, eps: c.eps) }
            }
            if let dense = layer.dense {
                dense.forward(nrm, count: s, into: tmp)
            } else {
                try moe(layer, layer: l, nrm, count: s, into: tmp)
            }
            for i in 0..<(s * d) { x[i] += tmp[i] }
        }
        cache.appendTokens(tokens)

        let th = monotonicSeconds()
        var logits = [Float](repeating: 0, count: c.vocab)
        let last = FloatPtr.allocate(capacity: d)
        defer { last.deallocate() }
        finalNorm.withUnsafeBufferPointer { rmsNorm(last, x + (s - 1) * d, $0.baseAddress!, d, eps: c.eps) }
        logits.withUnsafeMutableBufferPointer { lmHead.matmul(last, count: 1, into: $0.baseAddress!) }
        lastTiming.head = monotonicSeconds() - th
        return logits
    }

    /// MLA attention for `s` rows at positions pos0..<pos0+s. Appends each row's latent
    /// and rotary key to the cache, then attends causally.
    func attention(_ w: AttentionWeights, layer l: Int, _ x: ConstFloatPtr, count s: Int, pos0: Int,
                   cache: KVCache, into out: FloatPtr) {
        let c = config, h = c.heads, nope = c.qkNope, rope = c.qkRope, qkh = c.qkHead
        let lora = c.kvLora, vh = c.vHead, kvRow = nope + vh

        // Queries [s, H*qkHead].
        let q = FloatPtr.allocate(capacity: s * h * qkh)
        defer { q.deallocate() }
        if let qA = w.qA {
            let qa = FloatPtr.allocate(capacity: s * c.qLora)
            defer { qa.deallocate() }
            qA.matmul(x, count: s, into: qa)
            w.qANorm.withUnsafeBufferPointer { nw in
                for i in 0..<s { rmsNorm(qa + i * c.qLora, qa + i * c.qLora, nw.baseAddress!, c.qLora, eps: c.eps) }
            }
            w.qB.matmul(qa, count: s, into: q)
        } else {
            w.qB.matmul(x, count: s, into: q)
        }

        // Compressed keys/values: normalise the latent, rotate the shared key, cache both.
        let kva = FloatPtr.allocate(capacity: s * (lora + rope))
        defer { kva.deallocate() }
        w.kvA.matmul(x, count: s, into: kva)
        w.kvANorm.withUnsafeBufferPointer { nw in
            for i in 0..<s {
                let row = kva + i * (lora + rope)
                rmsNorm(row, row, nw.baseAddress!, lora, eps: c.eps)
                ropeInterleave(row + lora, dim: rope, pos: pos0 + i, theta: c.ropeTheta)
                cache.latent[l].append(contentsOf: UnsafeBufferPointer(start: row, count: lora))
                cache.rope[l].append(contentsOf: UnsafeBufferPointer(start: row + lora, count: rope))
            }
        }

        // Per-head attention; heads run in parallel, each writing its own slice.
        let heads = FloatPtr.allocate(capacity: s * h * vh)
        defer { heads.deallocate() }
        let scale = c.attnScale
        cache.latent[l].withUnsafeBufferPointer { latBuf in
            cache.rope[l].withUnsafeBufferPointer { ropeBuf in
                let lat = latBuf.baseAddress!, kr = ropeBuf.baseAddress!
                for i in 0..<s {
                    let pos = pos0 + i
                    let n = pos + 1  // causal: keys 0...pos
                    parallelFor(h) { hd in
                        let qn = q + i * h * qkh + hd * qkh
                        var qr = [Float](repeating: 0, count: rope)
                        for j in 0..<rope { qr[j] = qn[nope + j] }
                        qr.withUnsafeMutableBufferPointer {
                            ropeInterleave($0.baseAddress!, dim: rope, pos: pos, theta: c.ropeTheta)
                        }
                        // Fold the no-rope query through this head's key rows of kv_b.
                        var qabs = [Float](repeating: 0, count: lora)
                        qabs.withUnsafeMutableBufferPointer {
                            w.kvB.accumulateTransposed(hd * kvRow ..< hd * kvRow + nope, coeff: qn, into: $0.baseAddress!)
                        }
                        var scores = [Float](repeating: 0, count: n)
                        qabs.withUnsafeBufferPointer { qa in
                            qr.withUnsafeBufferPointer { qrp in
                                for t in 0..<n {
                                    scores[t] = (dot(qa.baseAddress!, lat + t * lora, lora)
                                                 + dot(qrp.baseAddress!, kr + t * rope, rope)) * scale
                                }
                            }
                        }
                        var ctx = [Float](repeating: 0, count: lora)
                        scores.withUnsafeMutableBufferPointer { sp in
                            softmax(sp.baseAddress!, n)
                            ctx.withUnsafeMutableBufferPointer { cp in
                                for t in 0..<n { axpy(sp[t], lat + t * lora, cp.baseAddress!, lora) }
                            }
                        }
                        // Expand the attended latent through this head's value rows.
                        ctx.withUnsafeBufferPointer { cp in
                            w.kvB.matmulRows(hd * kvRow + nope ..< hd * kvRow + kvRow, cp.baseAddress!, count: 1,
                                             into: heads + i * h * vh + hd * vh, yStride: vh, yOffset: 0,
                                             parallel: false)
                        }
                    }
                }
            }
        }
        w.o.matmul(heads, count: s, into: out)
    }

    /// Picks experts for one row of router logits. Returns (expert ids, weights).
    func route(_ logits: ConstFloatPtr, bias: [Float]) -> ([Int], [Float]) {
        let c = config, e = c.experts
        let k = min(expertTopK ?? c.topK, c.topK)
        var score = [Float](repeating: 0, count: e)
        var choice = [Float](repeating: 0, count: e)
        for i in 0..<e { score[i] = sigmoid(logits[i]); choice[i] = score[i] + bias[i] }
        // Top-k by biased score (the bias steers load balance), weights from the raw score.
        var ids = Array(0..<e)
        ids.sort { choice[$0] > choice[$1] }
        var chosen = Array(ids.prefix(k))
        var weights = chosen.map { score[$0] }
        if expertTopP > 0 && expertTopP < 1 {
            let total = weights.reduce(0, +)
            var cum: Float = 0, keep = chosen.count
            for (j, wv) in weights.enumerated() {
                cum += wv
                if cum >= expertTopP * total { keep = j + 1; break }
            }
            chosen = Array(chosen.prefix(keep))
            weights = Array(weights.prefix(keep))
        }
        if c.normTopK {
            let sum = weights.reduce(0, +) + 1e-20
            weights = weights.map { $0 / sum }
        }
        return (chosen, weights.map { $0 * c.routedScale })
    }

    /// Mixture of experts for `s` rows: route, fetch the chosen experts (from the cache
    /// or SSD), run each expert on the rows that chose it, add the shared expert.
    func moe(_ layer: LayerWeights, layer l: Int, _ x: ConstFloatPtr, count s: Int, into out: FloatPtr) throws {
        let c = config, d = c.hidden, e = c.experts
        let logits = FloatPtr.allocate(capacity: s * e)
        defer { logits.deallocate() }
        layer.router!.matmul(x, count: s, into: logits)

        // Which rows go to which expert, with what weight.
        var assignments: [Int: [(row: Int, weight: Float)]] = [:]
        for i in 0..<s {
            let (ids, ws) = route(logits + i * e, bias: layer.routerBias)
            for (id, wv) in zip(ids, ws) { assignments[id, default: []].append((row: i, weight: wv)) }
        }
        let ids = assignments.keys.sorted()

        let tw = monotonicSeconds()
        let weights = try experts.fetch(layer: l, ids: ids)
        let tc = monotonicSeconds()
        lastTiming.expertWait += tc - tw

        // Shared expert first (writes `out`), routed experts accumulate on top.
        if let shared = layer.shared {
            shared.forward(x, count: s, into: out)
        } else {
            out.initialize(repeating: 0, count: s * d)
        }
        let inter = c.moeInter
        for id in ids {
            guard let w = weights[id], let rows = assignments[id] else { continue }
            let n = rows.count
            let xe = FloatPtr.allocate(capacity: n * d)
            let g = FloatPtr.allocate(capacity: n * inter), u = FloatPtr.allocate(capacity: n * inter)
            let y = FloatPtr.allocate(capacity: n * d)
            defer { xe.deallocate(); g.deallocate(); u.deallocate(); y.deallocate() }
            for (j, r) in rows.enumerated() { (xe + j * d).update(from: x + r.row * d, count: d) }
            w.gate.matmul(xe, count: n, into: g)
            w.up.matmul(xe, count: n, into: u)
            for i in 0..<(n * inter) { g[i] = silu(g[i]) * u[i] }
            w.down.matmul(g, count: n, into: y)
            for (j, r) in rows.enumerated() { axpy(r.weight, y + j * d, out + r.row * d, d) }
        }
        lastTiming.experts += monotonicSeconds() - tc
    }
}
