// How RAM is split between resident weights, the KV cache, and the expert cache, and
// what that means for disk traffic. Used by `coli plan`, `coli info` and at start-up.

import Foundation

public struct MemoryPlan: Sendable {
    public var ramBudget: Int64         // bytes the engine may use
    public var denseBytes: Int64        // resident (non-expert) weights
    public var kvBytes: Int64           // KV cache at full context
    public var overheadBytes: Int64     // activations, buffers, runtime
    public var expertBytes: Int64       // one routed expert on disk
    public var totalExperts: Int        // routed experts in the model

    /// Bytes left for cached experts.
    public var expertCache: Int64 { max(0, ramBudget - denseBytes - kvBytes - overheadBytes) }
    /// Experts that fit in the cache.
    public var cachedExperts: Int { expertBytes > 0 ? Int(expertCache / expertBytes) : 0 }
    /// Fraction of all experts that can be cached.
    public var coverage: Double { totalExperts > 0 ? min(1, Double(cachedExperts) / Double(totalExperts)) : 0 }

    /// KV bytes for `context` tokens: per layer, a latent and a rotary key per token.
    public static func kvBytes(_ c: ModelConfig, context: Int) -> Int64 {
        Int64(context) * Int64(c.layers) * Int64(c.kvLora + c.qkRope) * 4
    }

    /// The default budget when `--ram` is not given: 85% of what the OS reports as
    /// available, leaving room for everything else on the machine.
    public static func defaultBudget() -> Int64 { Int64(Double(HostMemory.available) * 0.85) }

    public init(config c: ModelConfig, ramGB: Double, denseBytes: Int64, expertBytes: Int64, context: Int) {
        ramBudget = ramGB > 0 ? Int64(ramGB * 1e9) : Self.defaultBudget()
        self.denseBytes = denseBytes
        kvBytes = Self.kvBytes(c, context: context)
        overheadBytes = 1_000_000_000
        self.expertBytes = expertBytes
        totalExperts = c.experts * max(0, c.layers - c.firstDense)
    }

    /// Human-readable summary.
    public func describe() -> String {
        """
        RAM budget        \(formatBytes(ramBudget))
          resident        \(formatBytes(denseBytes))
          KV cache        \(formatBytes(kvBytes))
          overhead        \(formatBytes(overheadBytes))
          expert cache    \(formatBytes(expertCache))  (\(cachedExperts) of \(totalExperts) experts, \(String(format: "%.1f", coverage * 100))%)
        expert size       \(formatBytes(expertBytes))
        """
    }

    public var json: [String: Any] {
        ["ram_budget": ramBudget, "dense_bytes": denseBytes, "kv_bytes": kvBytes,
         "overhead_bytes": overheadBytes, "expert_cache_bytes": expertCache,
         "expert_bytes": expertBytes, "cached_experts": cachedExperts,
         "total_experts": totalExperts, "coverage": coverage]
    }
}
