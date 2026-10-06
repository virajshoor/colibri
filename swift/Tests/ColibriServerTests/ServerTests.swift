// Tests for the HTTP parser, the OpenAI request decoder, and the server's routing,
// using a fake backend so no model is needed.

import ColibriCore
import Foundation
import Testing
@testable import ColibriServer

@Test func parsesHTTPRequests() {
    let raw = Array("POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: localhost:8000\r\nContent-Length: 2\r\n\r\n{}".utf8)
    guard case .complete(let r) = HTTP.parse(raw, maxBody: 100) else {
        Issue.record("expected a complete request")
        return
    }
    #expect(r.method == "POST")
    #expect(r.path == "/v1/chat/completions")
    #expect(r.query == "x=1")
    #expect(r.header("HOST") == "localhost:8000")
    #expect(r.body == Data("{}".utf8))
    #expect(HTTP.parse(Array(raw.dropLast()), maxBody: 100) == .incomplete)
    #expect(HTTP.parse(Array("GET / HTTP/1.1\r\nHost: a".utf8), maxBody: 100) == .incomplete)
    if case .failure(let status, _) = HTTP.parse(raw, maxBody: 1) { #expect(status == 413) } else { Issue.record("413") }
    let chunked = Array("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n".utf8)
    if case .failure(let status, _) = HTTP.parse(chunked, maxBody: 100) { #expect(status == 411) } else { Issue.record("411") }
    if case .failure(let status, _) = HTTP.parse(Array("NONSENSE\r\n\r\n".utf8), maxBody: 100) {
        #expect(status == 400)
    } else { Issue.record("400") }
}

@Test func decodesOpenAIRequests() throws {
    let body = """
        {"model":"x","messages":[{"role":"system","content":"be brief"},
         {"role":"user","content":[{"type":"text","text":"hi"},{"type":"text","text":"there"}]}],
         "max_tokens":12,"temperature":0.2,"top_p":0.5,"stop":"END","stream":true,
         "stream_options":{"include_usage":true},"seed":7,"chat_template_kwargs":{"enable_thinking":false}}
        """
    let r = try OpenAI.decode(Data(body.utf8), kind: .chat)
    #expect(r.messages == [ChatMessage(role: "system", content: "be brief"), ChatMessage(role: "user", content: "hi\nthere")])
    #expect(r.options.sampling.maxTokens == 12)
    #expect(r.options.sampling.temperature == 0.2)
    #expect(r.options.sampling.topP == 0.5)
    #expect(r.options.sampling.seed == 7)
    #expect(r.options.stop == ["END"])
    #expect(r.stream && r.includeUsage)
    #expect(r.thinking == false)

    let t = try OpenAI.decode(Data(#"{"prompt":"abc","stop":["x","y"]}"#.utf8), kind: .text)
    #expect(t.prompt == "abc" && t.options.stop == ["x", "y"] && !t.stream)

    #expect(throws: OpenAIError.self) { try OpenAI.decode(Data(#"{"messages":[]}"#.utf8), kind: .chat) }
    #expect(throws: OpenAIError.self) { try OpenAI.decode(Data(#"{"prompt":"a","temperature":9}"#.utf8), kind: .text) }
    #expect(throws: OpenAIError.self) { try OpenAI.decode(Data("not json".utf8), kind: .text) }
}

/// Echoes the last user message back in two pieces.
final class FakeBackend: CompletionBackend {
    let modelID = "tiny"
    func run(_ request: CompletionRequest, shouldContinue: () -> Bool, onText: (String) -> Void) throws
        -> GenerationResult
    {
        let text = request.kind == .chat ? (request.messages.last?.content ?? "") : request.prompt
        var r = GenerationResult()
        r.promptTokens = 3
        for piece in [String(text.prefix(2)), String(text.dropFirst(2))] where !piece.isEmpty {
            guard shouldContinue() else { r.finishReason = .cancelled; return r }
            onText(piece)
            r.text += piece
            r.completionTokens += 1
        }
        return r
    }
    func expertMap() -> [String: Any] { ["rows": 1, "cols": 1, "map": "00"] }
}

/// Runs one request through the server's router and returns the raw response.
func roundTrip(_ server: OpenAIServer, _ method: String, _ path: String, body: String = "",
               headers: [String: String] = ["host": "localhost:8000"]) -> String {
    var out = Data()
    let r = HTTPRequest(method: method, path: path, query: "", headers: headers, body: Data(body.utf8))
    server.respond(r, write: { out.append($0); return true }, clientGone: { false })
    return String(decoding: out, as: UTF8.self)
}

@Test func serverRoutes() throws {
    let server = OpenAIServer(config: ServerConfig(), backend: FakeBackend())

    let health = roundTrip(server, "GET", "/health")
    #expect(health.hasPrefix("HTTP/1.1 200"))
    #expect(health.contains(#""model":"tiny""#))
    #expect(roundTrip(server, "GET", "/v1/models").contains(#""id":"tiny""#))
    #expect(roundTrip(server, "GET", "/nope").hasPrefix("HTTP/1.1 404"))

    let chat = roundTrip(server, "POST", "/v1/chat/completions",
                         body: #"{"messages":[{"role":"user","content":"hello"}]}"#)
    #expect(chat.hasPrefix("HTTP/1.1 200"))
    #expect(chat.contains(#""content":"hello""#))
    #expect(chat.contains(#""chat.completion""#))
    #expect(chat.contains(#""colibri":"#))

    let stream = roundTrip(server, "POST", "/v1/chat/completions",
                           body: #"{"messages":[{"role":"user","content":"hello"}],"stream":true,"stream_options":{"include_usage":true}}"#)
    #expect(stream.contains("text/event-stream"))
    #expect(stream.contains(#""content":"he""#))
    #expect(stream.contains(#""content":"llo""#))
    #expect(stream.contains(#""finish_reason":"stop""#))
    #expect(stream.contains(#""usage":"#))
    #expect(stream.hasSuffix("data: [DONE]\n\n"))

    let text = roundTrip(server, "POST", "/v1/completions", body: #"{"prompt":"abc"}"#)
    #expect(text.contains(#""text":"abc""#))

    let bad = roundTrip(server, "POST", "/v1/completions", body: "{}")
    #expect(bad.hasPrefix("HTTP/1.1 400"))
}

@Test func serverSecurity() {
    var c = ServerConfig()
    c.apiKey = "secret"
    c.corsOrigins = ["https://app.example"]
    let server = OpenAIServer(config: c, backend: FakeBackend())
    #expect(roundTrip(server, "GET", "/v1/models").hasPrefix("HTTP/1.1 401"))
    #expect(roundTrip(server, "GET", "/v1/models",
                      headers: ["host": "localhost", "authorization": "Bearer secret"]).hasPrefix("HTTP/1.1 200"))
    #expect(roundTrip(server, "GET", "/health").hasPrefix("HTTP/1.1 200"))  // health needs no key
    // DNS rebinding: a foreign Host is refused on a loopback bind.
    #expect(roundTrip(server, "GET", "/health", headers: ["host": "evil.example"]).hasPrefix("HTTP/1.1 421"))
    let pre = roundTrip(server, "OPTIONS", "/v1/chat/completions",
                        headers: ["host": "localhost", "origin": "https://app.example"])
    #expect(pre.hasPrefix("HTTP/1.1 204"))
    #expect(pre.contains("Access-Control-Allow-Origin: https://app.example"))
}
