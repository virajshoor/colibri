// Test fixtures: a safetensors writer, an int4 quantizer, and a tiny random GLM-style
// model written to a temporary directory, so the whole loading and streaming path runs
// without downloading anything.

import Foundation
@testable import ColibriCore

/// One tensor to write: dtype name, shape, raw bytes.
struct RawTensor {
    var dtype: String
    var shape: [Int]
    var bytes: [UInt8]

    static func f32(_ shape: [Int], _ v: [Float]) -> RawTensor {
        RawTensor(dtype: "F32", shape: shape, bytes: v.withUnsafeBytes { Array($0) })
    }
}

/// Writes a safetensors file: 8-byte header length, JSON header, data.
func writeSafetensors(_ tensors: [String: RawTensor], to url: URL, metadata: [String: String]? = nil) throws {
    var header: [String: Any] = [:]
    var data: [UInt8] = []
    for name in tensors.keys.sorted() {
        let t = tensors[name]!
        header[name] = ["dtype": t.dtype, "shape": t.shape, "data_offsets": [data.count, data.count + t.bytes.count]]
        data += t.bytes
    }
    if let metadata { header["__metadata__"] = metadata }
    let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    var out: [UInt8] = []
    var n = UInt64(json.count)
    for _ in 0..<8 { out.append(UInt8(n & 0xFF)); n >>= 8 }
    out += json
    out += data
    try Data(out).write(to: url)
}

/// Per-row int4 quantization as the converter does it: scale = max|w| / 7, values
/// stored as q + 8, two per byte, low nibble first.
func quantizeInt4Row(_ w: [Float], rows: Int, cols: Int) -> (packed: [UInt8], scales: [Float]) {
    let rb = (cols + 1) / 2
    var packed = [UInt8](repeating: 0, count: rows * rb)
    var scales = [Float](repeating: 0, count: rows)
    for o in 0..<rows {
        let row = w[(o * cols)..<((o + 1) * cols)]
        let m = row.map { abs($0) }.max() ?? 0
        let s = m > 0 ? m / 7 : 1
        scales[o] = s
        for i in 0..<cols {
            let q = Int(max(-8, min(7, (w[o * cols + i] / s).rounded())))
            let nib = UInt8(q + 8)
            if i & 1 == 0 { packed[o * rb + i / 2] |= nib } else { packed[o * rb + i / 2] |= nib << 4 }
        }
    }
    return (packed, scales)
}

/// Deterministic pseudo-random floats in [-a, a].
struct TestRNG {
    var g = SeededRandom(seed: 42)
    mutating func values(_ n: Int, _ a: Float = 0.5) -> [Float] {
        (0..<n).map { _ in Float.random(in: -a...a, using: &g) }
    }
}

/// Dimensions of the tiny model.
enum Tiny {
    static let hidden = 32, layers = 3, heads = 2, experts = 8, topK = 2, moeInter = 16
    static let denseInter = 24, qLora = 16, kvLora = 16, nope = 8, rope = 4, vHead = 8, vocab = 50

    static var config: [String: Any] {
        ["model_type": "glm_moe_dsa", "hidden_size": hidden, "num_hidden_layers": layers,
         "num_attention_heads": heads, "n_routed_experts": experts, "num_experts_per_tok": topK,
         "moe_intermediate_size": moeInter, "intermediate_size": denseInter, "first_k_dense_replace": 1,
         "q_lora_rank": qLora, "kv_lora_rank": kvLora, "qk_nope_head_dim": nope, "qk_rope_head_dim": rope,
         "v_head_dim": vHead, "n_shared_experts": 1, "vocab_size": vocab, "norm_topk_prob": true,
         "rms_norm_eps": 1e-5, "routed_scaling_factor": 2.5, "rope_theta": 10000, "eos_token_id": [0],
         "n_group": 1]
    }

    /// Writes config.json and two shards (dense + experts) into a fresh directory.
    static func write() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("colibri-tiny-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: config).write(to: dir.appendingPathComponent("config.json"))

        var rng = TestRNG()
        var dense: [String: RawTensor] = [:]
        var expertsShard: [String: RawTensor] = [:]
        func f32(_ name: String, _ shape: [Int], scale: Float = 0.3) {
            dense[name] = .f32(shape, rng.values(shape.reduce(1, *), scale))
        }
        func ones(_ name: String, _ n: Int) { dense[name] = .f32([n], [Float](repeating: 1, count: n)) }
        func q4(_ name: String, _ o: Int, _ i: Int, into dict: inout [String: RawTensor]) {
            let (p, s) = quantizeInt4Row(rng.values(o * i, 0.3), rows: o, cols: i)
            dict[name] = RawTensor(dtype: "U8", shape: [o, (i + 1) / 2], bytes: p)
            dict[name + ".qs"] = .f32([o], s)
        }

        f32("model.embed_tokens.weight", [vocab, hidden], scale: 1)
        f32("lm_head.weight", [vocab, hidden])
        ones("model.norm.weight", hidden)
        for l in 0..<layers {
            let p = "model.layers.\(l)."
            ones(p + "input_layernorm.weight", hidden)
            ones(p + "post_attention_layernorm.weight", hidden)
            q4(p + "self_attn.q_a_proj.weight", qLora, hidden, into: &dense)
            ones(p + "self_attn.q_a_layernorm.weight", qLora)
            q4(p + "self_attn.q_b_proj.weight", heads * (nope + rope), qLora, into: &dense)
            q4(p + "self_attn.kv_a_proj_with_mqa.weight", kvLora + rope, hidden, into: &dense)
            ones(p + "self_attn.kv_a_layernorm.weight", kvLora)
            q4(p + "self_attn.kv_b_proj.weight", heads * (nope + vHead), kvLora, into: &dense)
            q4(p + "self_attn.o_proj.weight", hidden, heads * vHead, into: &dense)
            if l == 0 {
                q4(p + "mlp.gate_proj.weight", denseInter, hidden, into: &dense)
                q4(p + "mlp.up_proj.weight", denseInter, hidden, into: &dense)
                q4(p + "mlp.down_proj.weight", hidden, denseInter, into: &dense)
            } else {
                f32(p + "mlp.gate.weight", [experts, hidden])
                dense[p + "mlp.gate.e_score_correction_bias"] = .f32([experts], rng.values(experts, 0.05))
                q4(p + "mlp.shared_experts.gate_proj.weight", moeInter, hidden, into: &dense)
                q4(p + "mlp.shared_experts.up_proj.weight", moeInter, hidden, into: &dense)
                q4(p + "mlp.shared_experts.down_proj.weight", hidden, moeInter, into: &dense)
                for e in 0..<experts {
                    let ep = p + "mlp.experts.\(e)."
                    q4(ep + "gate_proj.weight", moeInter, hidden, into: &expertsShard)
                    q4(ep + "up_proj.weight", moeInter, hidden, into: &expertsShard)
                    q4(ep + "down_proj.weight", hidden, moeInter, into: &expertsShard)
                }
            }
        }
        try writeSafetensors(dense, to: dir.appendingPathComponent("model-00001-of-00002.safetensors"))
        try writeSafetensors(expertsShard, to: dir.appendingPathComponent("model-00002-of-00002.safetensors"))
        return dir
    }
}
