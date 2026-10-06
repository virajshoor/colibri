// Picks the next token from logits: greedy at temperature 0, otherwise temperature
// scaling, optional top-k, then nucleus (top-p) sampling.

import Foundation

/// SplitMix64: small, fast, and seedable so a request with `seed` is reproducible.
public struct SeededRandom: RandomNumberGenerator {
    var state: UInt64
    public init(seed: UInt64) { state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

public struct SamplingParameters: Sendable, Equatable {
    public var temperature: Float = 1.0
    public var topP: Float = 0.95
    public var topK: Int = 0  // 0 = no limit
    public var maxTokens: Int = 2048
    public var seed: UInt64? = nil
    public init() {}
}

public struct Sampler {
    public var params: SamplingParameters
    var rng: SeededRandom

    public init(_ params: SamplingParameters) {
        self.params = params
        rng = SeededRandom(seed: params.seed ?? UInt64.random(in: 0...UInt64.max))
    }

    /// Index of the largest logit.
    public static func argmax(_ logits: [Float]) -> Int {
        var best = 0
        for i in 1..<logits.count where logits[i] > logits[best] { best = i }
        return best
    }

    public mutating func sample(_ logits: [Float]) -> Int {
        if params.temperature <= 0 { return Self.argmax(logits) }
        // Candidates sorted by logit; only the top-k (or a generous 4096 cap that
        // covers any realistic nucleus) need softmax and sorting.
        let limit = params.topK > 0 ? min(params.topK, logits.count) : min(4096, logits.count)
        var idx = Array(0..<logits.count)
        if limit < logits.count {
            idx = Array(idx.sorted { logits[$0] > logits[$1] }.prefix(limit))
        } else {
            idx.sort { logits[$0] > logits[$1] }
        }
        let inv = 1 / params.temperature
        let m = logits[idx[0]]
        var probs = idx.map { expf((logits[$0] - m) * inv) }
        let total = probs.reduce(0, +)
        probs = probs.map { $0 / total }
        // Nucleus: smallest prefix whose probability mass reaches top_p.
        var keep = probs.count
        if params.topP > 0 && params.topP < 1 {
            var cum: Float = 0
            for (i, p) in probs.enumerated() {
                cum += p
                if cum >= params.topP { keep = i + 1; break }
            }
        }
        let mass = probs.prefix(keep).reduce(0, +)
        var r = Float.random(in: 0..<1, using: &rng) * mass
        for i in 0..<keep {
            r -= probs[i]
            if r <= 0 { return idx[i] }
        }
        return idx[keep - 1]
    }
}
