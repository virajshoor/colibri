// The OpenAI request and response shapes for /v1/chat/completions and
// /v1/completions: decoding a request body into engine inputs, and building the JSON
// objects for full responses and streaming chunks. No I/O here.

import ColibriCore
import Foundation

/// An error that becomes an OpenAI-style error response.
public struct OpenAIError: Error, Sendable {
    public var status: Int
    public var message: String
    public var type: String
    public init(status: Int, message: String, type: String = "invalid_request_error") {
        self.status = status; self.message = message; self.type = type
    }
    public var body: [String: Any] { ["error": ["message": message, "type": type, "code": NSNull()]] }
}

/// A decoded completion request.
public struct CompletionRequest: Sendable {
    public enum Kind: Sendable { case chat, text }
    public var kind: Kind
    public var messages: [ChatMessage] = []
    public var prompt = ""
    public var stream = false
    public var includeUsage = false
    public var thinking: Bool? = nil
    public var options = GenerationOptions()
    public var model: String? = nil
}

public enum OpenAI {
    /// Defaults applied when a request leaves a field out.
    public struct Defaults: Sendable {
        public var temperature: Float = 1.0
        public var topP: Float = 0.95
        public var maxTokens = 2048
        public init() {}
    }

    /// Decodes a request body. `kind` comes from the route.
    public static func decode(_ body: Data, kind: CompletionRequest.Kind, defaults: Defaults = Defaults()) throws
        -> CompletionRequest
    {
        let o = try JSON.decodeObject(body)
        var r = CompletionRequest(kind: kind)
        r.model = o["model"] as? String
        switch kind {
        case .chat:
            guard let msgs = o["messages"] as? [[String: Any]], !msgs.isEmpty else {
                throw OpenAIError(status: 400, message: "`messages` must be a non-empty array")
            }
            r.messages = try msgs.map { (m: [String: Any]) throws -> ChatMessage in
                guard let role = m["role"] as? String else { throw OpenAIError(status: 400, message: "each message needs a `role`") }
                return ChatMessage(role: role, content: try text(of: m["content"]))
            }
        case .text:
            if let p = o["prompt"] as? String {
                r.prompt = p
            } else if let a = o["prompt"] as? [String], a.count == 1 {
                r.prompt = a[0]
            } else {
                throw OpenAIError(status: 400, message: "`prompt` must be a string")
            }
        }
        r.stream = (o["stream"] as? Bool) ?? false
        if let so = o["stream_options"] as? [String: Any] { r.includeUsage = (so["include_usage"] as? Bool) ?? false }

        var s = SamplingParameters()
        s.temperature = try number(o, "temperature", defaults.temperature, 0...2)
        s.topP = try number(o, "top_p", defaults.topP, 0...1)
        s.maxTokens = defaults.maxTokens
        for key in ["max_completion_tokens", "max_tokens"] {
            if let n = o[key] as? NSNumber {
                guard n.intValue > 0 else { throw OpenAIError(status: 400, message: "`\(key)` must be positive") }
                s.maxTokens = n.intValue
                break
            }
        }
        if let k = o["top_k"] as? NSNumber { s.topK = max(0, k.intValue) }
        if let seed = o["seed"] as? NSNumber { s.seed = UInt64(bitPattern: seed.int64Value) }
        r.options.sampling = s

        if let stop = o["stop"] as? String {
            r.options.stop = stop.isEmpty ? [] : [stop]
        } else if let stops = o["stop"] as? [String] {
            guard stops.count <= 16 else { throw OpenAIError(status: 400, message: "at most 16 stop sequences") }
            r.options.stop = stops.filter { !$0.isEmpty }
        }

        // Reasoning toggle: chat_template_kwargs.enable_thinking (vLLM/SGLang style),
        // or reasoning_effort ("none"/"minimal" turn it off).
        if let kw = o["chat_template_kwargs"] as? [String: Any], let t = kw["enable_thinking"] as? Bool {
            r.thinking = t
        } else if let effort = o["reasoning_effort"] as? String {
            r.thinking = !(effort == "none" || effort == "minimal")
        }
        return r
    }

    /// Message content may be a string or an array of parts; text parts are joined.
    static func text(of content: Any?) throws -> String {
        if content == nil || content is NSNull { return "" }
        if let s = content as? String { return s }
        if let parts = content as? [[String: Any]] {
            var out: [String] = []
            for p in parts {
                let type = p["type"] as? String ?? "text"
                guard type == "text" || type == "input_text" else {
                    throw OpenAIError(status: 400, message: "content part type `\(type)` is not supported by this model")
                }
                out.append(p["text"] as? String ?? "")
            }
            return out.joined(separator: "\n")
        }
        throw OpenAIError(status: 400, message: "message `content` must be a string or an array of text parts")
    }

    static func number(_ o: [String: Any], _ key: String, _ def: Float, _ range: ClosedRange<Float>) throws -> Float {
        guard let v = o[key], !(v is NSNull) else { return def }
        guard let n = v as? NSNumber, range.contains(n.floatValue) else {
            throw OpenAIError(status: 400, message: "`\(key)` must be a number in \(range.lowerBound)...\(range.upperBound)")
        }
        return n.floatValue
    }

    // MARK: responses

    public static func usage(_ r: GenerationResult) -> [String: Any] {
        ["prompt_tokens": r.promptTokens, "completion_tokens": r.completionTokens,
         "total_tokens": r.promptTokens + r.completionTokens,
         "prompt_tokens_details": ["cached_tokens": r.reusedTokens]]
    }

    /// colibrì's extra statistics, attached to every response.
    public static func stats(_ r: GenerationResult) -> [String: Any] {
        let e = r.experts
        return ["tokens_per_second": r.tokensPerSecond, "prefill_seconds": r.prefillSeconds,
                "decode_seconds": r.decodeSeconds, "expert_hit_rate": e.hitRate,
                "expert_hits": e.hits, "expert_misses": e.misses, "expert_bytes_read": e.bytesRead,
                "expert_read_seconds": e.readSeconds, "cached_experts": e.cachedExperts,
                "cached_expert_bytes": e.cachedBytes, "pinned_experts": e.pinnedExperts]
    }

    static func finish(_ r: GenerationResult) -> String {
        r.finishReason == .length ? "length" : "stop"
    }

    public static func chatResponse(id: String, model: String, created: Int, result r: GenerationResult) -> [String: Any] {
        ["id": id, "object": "chat.completion", "created": created, "model": model,
         "choices": [["index": 0, "message": ["role": "assistant", "content": r.text],
                      "finish_reason": finish(r), "logprobs": NSNull()]],
         "usage": usage(r), "colibri": stats(r)]
    }

    public static func textResponse(id: String, model: String, created: Int, result r: GenerationResult) -> [String: Any] {
        ["id": id, "object": "text_completion", "created": created, "model": model,
         "choices": [["index": 0, "text": r.text, "finish_reason": finish(r), "logprobs": NSNull()]],
         "usage": usage(r), "colibri": stats(r)]
    }

    /// One streaming chunk. `delta` is the new text (nil for the role or final chunk).
    public static func chunk(kind: CompletionRequest.Kind, id: String, model: String, created: Int,
                             delta: String?, role: Bool = false, finish: String? = nil) -> [String: Any] {
        let fr: Any = finish ?? NSNull()
        switch kind {
        case .chat:
            var d: [String: Any] = [:]
            if role { d["role"] = "assistant" }
            if let delta { d["content"] = delta }
            return ["id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [["index": 0, "delta": d, "finish_reason": fr, "logprobs": NSNull()]]]
        case .text:
            return ["id": id, "object": "text_completion", "created": created, "model": model,
                    "choices": [["index": 0, "text": delta ?? "", "finish_reason": fr, "logprobs": NSNull()]]]
        }
    }

    public static func usageChunk(kind: CompletionRequest.Kind, id: String, model: String, created: Int,
                                  result r: GenerationResult) -> [String: Any] {
        ["id": id, "object": kind == .chat ? "chat.completion.chunk" : "text_completion",
         "created": created, "model": model, "choices": [] as [Any], "usage": usage(r)]
    }

    public static func models(id: String, created: Int) -> [String: Any] {
        ["object": "list", "data": [["id": id, "object": "model", "created": created, "owned_by": "colibri"]]]
    }
}
