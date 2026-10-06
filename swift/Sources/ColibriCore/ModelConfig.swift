// Model hyper-parameters from config.json (and stop tokens from generation_config.json)
// for the GLM / DeepSeek-V3 family: multi-head latent attention (MLA) plus a
// mixture-of-experts feed-forward with sigmoid routing and a shared expert.
//
// Every value is range-checked: config.json often comes from an untrusted mirror and
// these numbers size allocations.

import Foundation

public struct ModelConfig: Sendable {
    public var hidden = 0          // model width D
    public var layers = 0
    public var heads = 0
    public var experts = 0         // routed experts per sparse layer
    public var topK = 0            // experts chosen per token
    public var moeInter = 0        // routed expert hidden width
    public var denseInter = 0      // dense MLP hidden width (first layers)
    public var firstDense = 0      // layers [0, firstDense) use a dense MLP
    public var qLora = 0           // query low-rank width; 0 = direct q_proj
    public var kvLora = 0          // compressed KV latent width
    public var qkNope = 0          // per-head query/key width without rotary
    public var qkRope = 0          // per-head rotary width (shared key across heads)
    public var vHead = 0           // per-head value width
    public var sharedExperts = 0
    public var vocab = 0
    public var normTopK = false    // renormalise the chosen experts' weights to sum 1
    public var eps: Float = 1e-5
    public var ropeTheta: Float = 10000
    public var routedScale: Float = 1
    public var indexTopK = 0       // DSA indexer window (attention is exact up to this)
    public var stopTokens: [Int] = []
    public var modelType = ""

    public var qkHead: Int { qkNope + qkRope }
    public var attnScale: Float { 1 / Float(qkHead).squareRoot() }

    public init() {}

    /// Reads `config.json` and, when present, merges `generation_config.json`'s stop
    /// tokens (Hugging Face treats that file as the authority for generation).
    public static func load(directory: URL) throws -> ModelConfig {
        let url = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url) else { throw ColibriError("cannot read \(url.path)") }
        guard var root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ColibriError("\(url.path) is not a JSON object")
        }
        // Multimodal checkpoints nest the language model's settings.
        if let text = root["text_config"] as? [String: Any] { root.merge(text) { _, new in new } }
        var c = try ModelConfig(json: root)
        let gen = directory.appendingPathComponent("generation_config.json")
        if let gd = try? Data(contentsOf: gen),
           let g = try? JSONSerialization.jsonObject(with: gd) as? [String: Any] {
            for id in Self.ints(g["eos_token_id"]) where !c.stopTokens.contains(id) { c.stopTokens.append(id) }
        }
        return c
    }

    /// Builds a configuration from a parsed config.json object.
    public init(json r: [String: Any]) throws {
        func int(_ k: String) -> Int { (r[k] as? NSNumber)?.intValue ?? 0 }
        func float(_ k: String, _ d: Float) -> Float { (r[k] as? NSNumber)?.floatValue ?? d }
        hidden = int("hidden_size")
        layers = int("num_hidden_layers")
        heads = int("num_attention_heads")
        experts = int("n_routed_experts")
        topK = int("num_experts_per_tok")
        moeInter = int("moe_intermediate_size")
        denseInter = int("intermediate_size")
        firstDense = int("first_k_dense_replace")
        qLora = int("q_lora_rank")
        kvLora = int("kv_lora_rank")
        qkNope = int("qk_nope_head_dim")
        qkRope = int("qk_rope_head_dim")
        vHead = int("v_head_dim")
        sharedExperts = int("n_shared_experts")
        vocab = int("vocab_size")
        normTopK = (r["norm_topk_prob"] as? Bool) ?? false
        eps = float("rms_norm_eps", 1e-5)
        routedScale = float("routed_scaling_factor", 1)
        if let rp = r["rope_parameters"] as? [String: Any], let t = rp["rope_theta"] as? NSNumber {
            ropeTheta = t.floatValue
        } else {
            ropeTheta = float("rope_theta", 10000)
        }
        indexTopK = int("index_topk")
        stopTokens = Self.ints(r["eos_token_id"])
        modelType = (r["model_type"] as? String) ?? ""
        if let g = r["n_group"] as? NSNumber, g.intValue > 1 {
            throw ColibriError("config: n_group=\(g) (grouped routing) is not supported; GLM-5 uses n_group=1")
        }
        try validate()
    }

    /// `eos_token_id` may be a number or a list of numbers.
    static func ints(_ v: Any?) -> [Int] {
        if let n = v as? NSNumber { return [n.intValue] }
        if let a = v as? [Any] { return a.compactMap { ($0 as? NSNumber)?.intValue } }
        return []
    }

    func validate() throws {
        func check(_ name: String, _ v: Int, _ lo: Int, _ hi: Int) throws {
            if v < lo || v > hi { throw ColibriError("config: \(name)=\(v) is outside [\(lo), \(hi)]") }
        }
        try check("hidden_size", hidden, 1, 1 << 20)
        try check("num_hidden_layers", layers, 1, 256)
        try check("num_attention_heads", heads, 1, 1024)
        try check("n_routed_experts", experts, 1, 4096)
        try check("num_experts_per_tok", topK, 1, min(64, experts))
        try check("moe_intermediate_size", moeInter, 1, 1 << 20)
        try check("intermediate_size", denseInter, 0, 1 << 24)
        try check("first_k_dense_replace", firstDense, 0, layers)
        try check("q_lora_rank", qLora, 0, 1 << 20)
        try check("kv_lora_rank", kvLora, 1, 1 << 20)
        try check("qk_nope_head_dim", qkNope, 1, 1 << 16)
        try check("qk_rope_head_dim", qkRope, 2, 1 << 16)
        try check("v_head_dim", vHead, 1, 1 << 16)
        try check("n_shared_experts", sharedExperts, 0, 64)
        try check("vocab_size", vocab, 1, 1 << 24)
        if qkRope % 2 != 0 { throw ColibriError("config: qk_rope_head_dim must be even") }
        if firstDense > 0 && denseInter < 1 { throw ColibriError("config: dense layers need intermediate_size") }
    }
}
