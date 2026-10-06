// Text generation on top of the model: prompt prefill with KV-cache reuse, the decode
// loop, stop tokens and stop strings, streaming text, and per-turn statistics.
//
// One Engine owns one KV cache (one conversation slot). Requests are serialised by a
// lock; a new prompt that starts with what the cache already holds (the usual case for
// chat, where each turn resends the history) only prefills the new suffix.

import Foundation

/// A chat message for the template.
public struct ChatMessage: Sendable, Equatable {
    public var role: String
    public var content: String
    public init(role: String, content: String) { self.role = role; self.content = content }
}

/// Tokenizer and chat template, provided by the front end (the `coli` command wraps
/// swift-transformers). Kept as a protocol so the engine and server test without one.
public protocol TextTokenizer: AnyObject {
    func encode(_ text: String) -> [Int]
    func decode(_ tokens: [Int]) -> String
    /// Renders messages with the model's chat template, ending with the assistant
    /// generation prompt. `thinking` toggles reasoning for templates that support it.
    func chatPrompt(_ messages: [ChatMessage], thinking: Bool?) throws -> [Int]
    /// End-of-sequence tokens known to the tokenizer.
    var eosTokens: [Int] { get }
}

/// Why generation ended, in OpenAI's vocabulary.
public enum FinishReason: String, Sendable {
    case stop, length, cancelled
}

public struct GenerationResult: Sendable {
    public var text = ""
    public var promptTokens = 0
    public var reusedTokens = 0       // prompt tokens served from the KV cache
    public var completionTokens = 0
    public var finishReason = FinishReason.stop
    public var prefillSeconds = 0.0
    public var decodeSeconds = 0.0
    public var experts = ExpertStats()
    public var tokensPerSecond: Double { decodeSeconds > 0 ? Double(completionTokens) / decodeSeconds : 0 }
    public init() {}
}

/// Generation knobs that are not sampling parameters.
public struct GenerationOptions: Sendable {
    public var sampling = SamplingParameters()
    public var stop: [String] = []
    public init() {}
}

public final class Engine: @unchecked Sendable {
    public let model: GLMModel
    public let tokenizer: TextTokenizer
    /// Largest context (prompt + reply) accepted.
    public let maxContext: Int
    /// Tokens per prefill chunk; bounds activation memory for long prompts.
    public var prefillChunk = 256
    /// Persist `.coli_usage` after each turn.
    public var saveUsage = true

    let cache: KVCache
    let lock = NSLock()
    let stopTokens: Set<Int>

    public init(model: GLMModel, tokenizer: TextTokenizer, maxContext: Int) {
        self.model = model
        self.tokenizer = tokenizer
        self.maxContext = maxContext
        cache = KVCache(config: model.config)
        stopTokens = Set(model.config.stopTokens + tokenizer.eosTokens)
    }

    /// Clears the conversation held in the KV cache.
    public func reset() {
        lock.lock(); defer { lock.unlock() }
        cache.truncate(to: 0)
    }

    /// Generates a reply to `prompt` (already tokenized). `onText` receives text as it
    /// becomes final; returning false from `shouldContinue` cancels.
    public func generate(prompt: [Int], options: GenerationOptions,
                         shouldContinue: () -> Bool = { true },
                         onText: (String) -> Void = { _ in }) throws -> GenerationResult {
        lock.lock(); defer { lock.unlock() }
        guard !prompt.isEmpty else { throw ColibriError("empty prompt") }
        guard prompt.count < maxContext else {
            throw ColibriError("prompt is \(prompt.count) tokens; the context limit is \(maxContext)")
        }
        var result = GenerationResult()
        result.promptTokens = prompt.count
        let maxNew = min(options.sampling.maxTokens, maxContext - prompt.count)

        // Reuse the longest common prefix; at least one prompt token must run so the
        // model produces fresh logits.
        var common = 0
        let held = cache.tokens
        while common < held.count && common < prompt.count && held[common] == prompt[common] { common += 1 }
        common = min(common, prompt.count - 1)
        cache.truncate(to: common)
        result.reusedTokens = common

        let t0 = monotonicSeconds()
        var logits: [Float] = []
        var done = common
        while done < prompt.count {
            let end = min(prompt.count, done + max(1, prefillChunk))
            logits = try model.forward(Array(prompt[done..<end]), cache: cache)
            done = end
            if !shouldContinue() {
                result.finishReason = .cancelled
                return finish(result)
            }
        }
        result.prefillSeconds = monotonicSeconds() - t0

        var sampler = Sampler(options.sampling)
        var generated: [Int] = []
        var emitted = ""        // text already handed to onText
        var reason = FinishReason.length
        var stoppedByString = false
        let td = monotonicSeconds()
        let longestStop = options.stop.map(\.count).max() ?? 0
        decode: while generated.count < maxNew {
            let next = sampler.sample(logits)
            if stopTokens.contains(next) { reason = .stop; break }
            generated.append(next)
            var text = tokenizer.decode(generated)
            // A token can end inside a multi-byte character; wait for the rest.
            if text.hasSuffix("\u{FFFD}") && generated.count < maxNew {
                logits = try model.forward([next], cache: cache)
                continue
            }
            if let hit = options.stop.compactMap({ text.range(of: $0) }).min(by: { $0.lowerBound < $1.lowerBound }) {
                text = String(text[..<hit.lowerBound])
                if text.count > emitted.count { onText(String(text.dropFirst(emitted.count))) }
                emitted = text
                reason = .stop
                stoppedByString = true
                break decode
            }
            // Hold back a tail that could still grow into a stop string.
            let safe = max(emitted.count, text.count - max(0, longestStop - 1))
            if safe > emitted.count {
                let chunk = String(text.dropFirst(emitted.count).prefix(safe - emitted.count))
                onText(chunk)
                emitted += chunk
            }
            if !shouldContinue() { reason = .cancelled; break }
            if generated.count >= maxNew { break }
            logits = try model.forward([next], cache: cache)
        }
        if !stoppedByString {
            // Flush the tail held back for stop-string matching.
            let text = tokenizer.decode(generated)
            if text.count > emitted.count { onText(String(text.dropFirst(emitted.count))) }
            emitted = text
        }
        result.decodeSeconds = monotonicSeconds() - td
        result.completionTokens = generated.count
        result.text = emitted
        result.finishReason = reason
        return finish(result)
    }

    private func finish(_ r: GenerationResult) -> GenerationResult {
        var r = r
        r.experts = model.experts.snapshot()
        if saveUsage {
            UsageHistory.save(model.experts.usageSnapshot(), directory: model.directory)
        }
        return r
    }

    /// Convenience: render chat messages and generate.
    public func chat(_ messages: [ChatMessage], thinking: Bool? = nil, options: GenerationOptions,
                     shouldContinue: () -> Bool = { true },
                     onText: (String) -> Void = { _ in }) throws -> GenerationResult {
        let prompt = try tokenizer.chatPrompt(messages, thinking: thinking)
        return try generate(prompt: prompt, options: options, shouldContinue: shouldContinue, onText: onText)
    }
}
