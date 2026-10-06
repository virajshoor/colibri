// The `.coli_usage` expert history: how often each (layer, expert) was routed to,
// accumulated across sessions in the model directory. At start-up the most used experts
// are pinned in RAM, so the cache "learns" the workload.
//
// The file format is the C engine's (see c/route_trace.h), so either engine can read
// the other's history:
//
//     -1 <n_layers> <n_experts>          dimensions
//     -2 <format_version> <engine_id>    writer identity (FNV-1a 32 of the engine name)
//     <layer> <expert> <count>           one line per non-zero counter
//
// The C engine keeps one extra row for its MTP layer; this reader accepts and ignores
// records for it.

import Foundation

public enum UsageHistory {
    public static let fileName = ".coli_usage"
    public static let formatVersion = 1
    /// Engine name the GLM path writes under; must match the C engine's for sharing.
    public static let engineName = "glm_moe_dsa"

    /// FNV-1a 32-bit hash, the C engine's `rt_hash`.
    public static func fnv1a(_ s: String) -> UInt32 {
        var h: UInt32 = 2_166_136_261
        for b in s.utf8 { h ^= UInt32(b); h = h &* 16_777_619 }
        return h
    }

    /// Parses a history into counters of shape [layers][experts]. Records outside that
    /// shape are skipped. Throws when the header declares other dimensions or a newer
    /// format, rather than misreading the file.
    public static func parse(_ text: String, layers: Int, experts: Int) throws -> [[UInt32]] {
        var out = Array(repeating: Array(repeating: UInt32(0), count: experts), count: layers)
        for line in text.split(whereSeparator: \.isNewline) {
            let f = line.split(separator: " ")
            guard f.count == 3, let a = Int(f[0]), let b = Int(f[1]), let c = UInt32(f[2]) else { break }
            switch a {
            case -1:
                // The C engine counts its MTP row, so n_layers may be ours or ours+1.
                if (b != layers && b != layers - 1 && b != layers + 1) || Int(c) != experts {
                    throw ColibriError("usage history is \(b) layers x \(c) experts, model is \(layers) x \(experts)")
                }
            case -2:
                if b > formatVersion { throw ColibriError("usage history format \(b) is newer than this build reads (\(formatVersion))") }
            case _ where a < 0:
                continue
            default:
                if a < layers && b >= 0 && b < experts { out[a][b] = out[a][b] &+ c }
            }
        }
        return out
    }

    /// Renders counters in the history format. An all-zero history is an empty file,
    /// like the C engine writes.
    public static func render(_ counts: [[UInt32]]) -> String {
        let nonZero = counts.contains { $0.contains { $0 != 0 } }
        guard nonZero, let experts = counts.first?.count else { return "" }
        var s = "-1 \(counts.count) \(experts)\n-2 \(formatVersion) \(fnv1a(engineName))\n"
        for (l, row) in counts.enumerated() {
            for (e, c) in row.enumerated() where c != 0 { s += "\(l) \(e) \(c)\n" }
        }
        return s
    }

    public static func load(directory: URL, layers: Int, experts: Int) throws -> [[UInt32]]? {
        let url = directory.appendingPathComponent(fileName)
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        return try parse(text, layers: layers, experts: experts)
    }

    /// Writes atomically (temp file + rename) so a crash never leaves a half history.
    public static func save(_ counts: [[UInt32]], directory: URL) {
        let url = directory.appendingPathComponent(fileName)
        try? render(counts).write(to: url, atomically: true, encoding: .utf8)
    }

    /// (layer, expert) pairs sorted by descending count, for pinning.
    public static func ranking(_ counts: [[UInt32]]) -> [(layer: Int, expert: Int)] {
        var all: [(Int, Int, UInt32)] = []
        for (l, row) in counts.enumerated() { for (e, c) in row.enumerated() where c > 0 { all.append((l, e, c)) } }
        all.sort { $0.2 > $1.2 }
        return all.map { (layer: $0.0, expert: $0.1) }
    }
}
