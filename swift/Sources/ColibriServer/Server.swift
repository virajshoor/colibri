// The OpenAI-compatible HTTP server behind `coli serve`, on plain POSIX sockets.
//
// Routes:
//   GET  /health                  liveness and model id (no auth)
//   GET  /v1/models               the served model
//   POST /v1/chat/completions     chat, streaming (SSE) or not
//   POST /v1/completions          raw text completion
//   GET  /experts                 expert cache map: tier and usage heat per expert
//   GET  /*                       static files from --web-root, if given
//
// Every response also carries a `colibri` object with throughput and expert-cache
// statistics; streams send it as one extra event before `data: [DONE]`.
//
// Each connection is handled on its own thread. The engine runs one request at a
// time; others wait in a bounded queue and get 503 when it is full.

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import ColibriCore
import Foundation

/// What the server needs from the model side.
public protocol CompletionBackend: AnyObject {
    var modelID: String { get }
    func run(_ request: CompletionRequest, shouldContinue: () -> Bool, onText: (String) -> Void) throws
        -> GenerationResult
    /// Payload of GET /experts.
    func expertMap() -> [String: Any]
}

/// Serves an `Engine`.
public final class EngineBackend: CompletionBackend {
    public let engine: Engine
    public let modelID: String

    public init(engine: Engine, modelID: String) {
        self.engine = engine
        self.modelID = modelID
    }

    public func run(_ r: CompletionRequest, shouldContinue: () -> Bool, onText: (String) -> Void) throws
        -> GenerationResult
    {
        switch r.kind {
        case .chat:
            return try engine.chat(r.messages, thinking: r.thinking, options: r.options,
                                   shouldContinue: shouldContinue, onText: onText)
        case .text:
            return try engine.generate(prompt: engine.tokenizer.encode(r.prompt), options: r.options,
                                       shouldContinue: shouldContinue, onText: onText)
        }
    }

    /// One byte per (layer, expert), hex-encoded, row-major: tier << 6 | heat, where
    /// tier is 1 for an expert in the RAM cache and 0 for one on disk, and heat is
    /// log2 of its usage count (0...63). Same encoding as the C engine's EMAP line.
    public func expertMap() -> [String: Any] {
        let m = engine.model, c = m.config
        let usage = m.experts.usageSnapshot()
        let digits = Array("0123456789abcdef")
        var hex = ""
        hex.reserveCapacity(c.layers * c.experts * 2)
        for l in 0..<c.layers {
            for e in 0..<c.experts {
                let n = usage[l][e]
                let heat = n == 0 ? 0 : min(63, 32 - n.leadingZeroBitCount)
                let tier = m.experts.isCached(layer: l, expert: e) ? 1 : 0
                let byte = tier << 6 | heat
                hex.append(digits[byte >> 4])
                hex.append(digits[byte & 15])
            }
        }
        let s = m.experts.snapshot()
        return ["rows": c.layers, "cols": c.experts, "first_sparse_layer": c.firstDense, "map": hex,
                "cached_experts": s.cachedExperts, "cached_bytes": s.cachedBytes,
                "pinned_experts": s.pinnedExperts, "hit_rate": s.hitRate]
    }
}

public struct ServerConfig: Sendable {
    public var host = "127.0.0.1"
    public var port = 8000
    public var apiKey: String? = nil
    public var corsOrigins: [String] = []
    /// Accepted Host header names. Empty means: only localhost names when bound to a
    /// loopback address (blocks DNS rebinding), anything otherwise.
    public var allowedHosts: [String] = []
    public var maxQueue = 8
    public var maxBodyBytes = 32 << 20
    public var webRoot: String? = nil
    public var defaults = OpenAI.Defaults()
    public init() {}
}

public final class OpenAIServer: @unchecked Sendable {
    let config: ServerConfig
    let backend: CompletionBackend
    let started = Int(Date().timeIntervalSince1970)
    private let queueLock = NSLock()
    private var inFlight = 0
    private var listenFD: Int32 = -1

    public init(config: ServerConfig, backend: CompletionBackend) {
        self.config = config
        self.backend = backend
    }

    // MARK: sockets

    /// Binds, listens, and serves forever on the calling thread.
    public func run(onListening: (String) -> Void = { _ in }) throws {
        signal(SIGPIPE, SIG_IGN)  // a client hanging up must not kill the server
        var hints = addrinfo()
        hints.ai_family = AF_UNSPEC
        #if canImport(Glibc)
        hints.ai_socktype = Int32(SOCK_STREAM.rawValue)
        #else
        hints.ai_socktype = SOCK_STREAM
        #endif
        hints.ai_flags = AI_PASSIVE
        var res: UnsafeMutablePointer<addrinfo>?
        let rc = getaddrinfo(config.host, String(config.port), &hints, &res)
        guard rc == 0, let ai = res else {
            throw ColibriError("cannot resolve \(config.host): \(String(cString: gai_strerror(rc)))")
        }
        defer { freeaddrinfo(res) }
        let fd = socket(ai.pointee.ai_family, ai.pointee.ai_socktype, ai.pointee.ai_protocol)
        guard fd >= 0 else { throw ColibriError("socket() failed") }
        var one: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, socklen_t(MemoryLayout<Int32>.size))
        guard bind(fd, ai.pointee.ai_addr, ai.pointee.ai_addrlen) == 0 else {
            close(fd)
            throw ColibriError("cannot bind \(config.host):\(config.port): \(String(cString: strerror(errno)))")
        }
        guard listen(fd, 128) == 0 else { close(fd); throw ColibriError("listen() failed") }
        listenFD = fd
        onListening("http://\(config.host.contains(":") ? "[\(config.host)]" : config.host):\(config.port)")
        while true {
            let client = accept(fd, nil, nil)
            if client < 0 {
                if errno == EINTR { continue }
                continue
            }
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                handle(client)
                close(client)
            }
        }
    }

    /// Sends all bytes; false if the peer is gone.
    @discardableResult
    func send(_ fd: Int32, _ data: Data) -> Bool {
        data.withUnsafeBytes { raw -> Bool in
            guard var p = raw.baseAddress else { return true }
            var left = raw.count
            while left > 0 {
                #if canImport(Glibc)
                let n = Glibc.send(fd, p, left, Int32(MSG_NOSIGNAL))
                #else
                let n = Darwin.send(fd, p, left, 0)
                #endif
                if n < 0 { if errno == EINTR { continue }; return false }
                if n == 0 { return false }
                p += n
                left -= n
            }
            return true
        }
    }

    /// True once the client has closed its side (it hung up while we generate).
    func peerClosed(_ fd: Int32) -> Bool {
        var pfd = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
        guard poll(&pfd, 1, 0) > 0 else { return false }
        if pfd.revents & Int16(POLLHUP | POLLERR) != 0 { return true }
        var b: UInt8 = 0
        let n = recv(fd, &b, 1, Int32(MSG_PEEK | MSG_DONTWAIT))
        return n == 0
    }

    /// Reads one request from the connection and answers it.
    func handle(_ fd: Int32) {
        #if canImport(Darwin)
        var one: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))
        #endif
        var tv = timeval(tv_sec: 30, tv_usec: 0)  // a client that stalls mid-request is dropped
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))

        var buf: [UInt8] = []
        var chunk = [UInt8](repeating: 0, count: 65536)
        var request: HTTPRequest?
        while request == nil {
            let n = chunk.withUnsafeMutableBytes { recv(fd, $0.baseAddress, $0.count, 0) }
            if n <= 0 { return }
            buf.append(contentsOf: chunk[0..<n])
            switch HTTP.parse(buf, maxBody: config.maxBodyBytes) {
            case .incomplete: continue
            case .failure(let status, let message):
                send(fd, errorResponse(OpenAIError(status: status, message: message), origin: nil))
                return
            case .complete(let r): request = r
            }
        }
        respond(fd, request!)
    }

    // MARK: routing

    func corsHeaders(_ origin: String?) -> [(String, String)] {
        guard let origin, config.corsOrigins.contains("*") || config.corsOrigins.contains(origin) else { return [] }
        return [("Access-Control-Allow-Origin", config.corsOrigins.contains("*") ? "*" : origin),
                ("Access-Control-Allow-Headers", "Authorization, Content-Type"),
                ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
                ("Vary", "Origin")]
    }

    func jsonResponse(_ status: Int, _ object: Any, origin: String?) -> Data {
        HTTP.response(status: status, headers: [("Content-Type", "application/json")] + corsHeaders(origin),
                      body: JSON.encode(object))
    }

    func errorResponse(_ e: OpenAIError, origin: String?) -> Data { jsonResponse(e.status, e.body, origin: origin) }

    /// Host header check against DNS rebinding: a web page on another domain must not
    /// be able to reach a server bound to localhost through the victim's browser.
    func hostAllowed(_ r: HTTPRequest) -> Bool {
        guard let host = r.header("host") else { return true }  // HTTP/1.0 clients
        var name = host.lowercased()
        if name.hasPrefix("[") { name = String(name.dropFirst().prefix { $0 != "]" }) }
        else if let colon = name.lastIndex(of: ":") { name = String(name[..<colon]) }
        if !config.allowedHosts.isEmpty { return config.allowedHosts.map { $0.lowercased() }.contains(name) }
        let loopback = ["127.0.0.1", "localhost", "::1"]
        if loopback.contains(config.host.lowercased()) { return loopback.contains(name) }
        return true
    }

    /// Constant-time comparison so the key cannot be guessed byte by byte from timing.
    func authorized(_ r: HTTPRequest) -> Bool {
        guard let key = config.apiKey, !key.isEmpty else { return true }
        let given = Array((r.header("authorization") ?? "").utf8)
        let want = Array("Bearer \(key)".utf8)
        guard given.count == want.count else { return false }
        var diff: UInt8 = 0
        for i in 0..<want.count { diff |= given[i] ^ want[i] }
        return diff == 0
    }

    /// Answers one request. Separated from the socket loop for testing: `write` gets
    /// every byte that would go to the client.
    public func respond(_ fd: Int32, _ r: HTTPRequest) {
        respond(r, write: { self.send(fd, $0) }, clientGone: { self.peerClosed(fd) })
    }

    public func respond(_ r: HTTPRequest, write: (Data) -> Bool, clientGone: () -> Bool) {
        let origin = r.header("origin")
        guard hostAllowed(r) else {
            _ = write(errorResponse(OpenAIError(status: 421, message: "Host not allowed"), origin: origin))
            return
        }
        if r.method == "OPTIONS" {
            _ = write(HTTP.response(status: 204, headers: corsHeaders(origin), body: Data()))
            return
        }
        let protected = r.path.hasPrefix("/v1/") || r.path == "/experts"
        if protected && !authorized(r) {
            _ = write(errorResponse(OpenAIError(status: 401, message: "invalid API key", type: "authentication_error"),
                                    origin: origin))
            return
        }
        switch (r.method, r.path) {
        case ("GET", "/health"):
            _ = write(jsonResponse(200, ["status": "ok", "model": backend.modelID], origin: origin))
        case ("GET", "/v1/models"):
            _ = write(jsonResponse(200, OpenAI.models(id: backend.modelID, created: started), origin: origin))
        case ("GET", "/v1/models/" + backend.modelID):
            _ = write(jsonResponse(200, (OpenAI.models(id: backend.modelID, created: started)["data"] as! [Any])[0],
                                   origin: origin))
        case ("GET", "/experts"):
            _ = write(jsonResponse(200, backend.expertMap(), origin: origin))
        case ("POST", "/v1/chat/completions"):
            complete(r, kind: .chat, origin: origin, write: write, clientGone: clientGone)
        case ("POST", "/v1/completions"):
            complete(r, kind: .text, origin: origin, write: write, clientGone: clientGone)
        case ("GET", _), ("HEAD", _):
            _ = write(staticFile(r.path, origin: origin))
        default:
            _ = write(errorResponse(OpenAIError(status: 404, message: "no route for \(r.method) \(r.path)"),
                                    origin: origin))
        }
    }

    /// Serves files from --web-root. Paths are resolved and must stay inside the root;
    /// unknown paths fall back to index.html so a single-page app can route.
    func staticFile(_ path: String, origin: String?) -> Data {
        guard let rootPath = config.webRoot else {
            return errorResponse(OpenAIError(status: 404, message: "not found"), origin: origin)
        }
        let root = URL(fileURLWithPath: rootPath).standardizedFileURL.resolvingSymlinksInPath()
        var file = root.appendingPathComponent(path == "/" ? "index.html" : String(path.dropFirst()))
            .standardizedFileURL.resolvingSymlinksInPath()
        guard file.path == root.path || file.path.hasPrefix(root.path + "/") else {
            return errorResponse(OpenAIError(status: 403, message: "forbidden"), origin: origin)
        }
        var isDir: ObjCBool = false
        if !FileManager.default.fileExists(atPath: file.path, isDirectory: &isDir) || isDir.boolValue {
            file = root.appendingPathComponent("index.html")
        }
        guard let data = try? Data(contentsOf: file) else {
            return errorResponse(OpenAIError(status: 404, message: "not found"), origin: origin)
        }
        let types = ["html": "text/html; charset=utf-8", "js": "text/javascript", "css": "text/css",
                     "json": "application/json", "svg": "image/svg+xml", "png": "image/png",
                     "ico": "image/x-icon", "woff2": "font/woff2"]
        let type = types[file.pathExtension.lowercased()] ?? "application/octet-stream"
        return HTTP.response(status: 200, headers: [("Content-Type", type)] + corsHeaders(origin), body: data)
    }

    // MARK: completions

    func complete(_ r: HTTPRequest, kind: CompletionRequest.Kind, origin: String?,
                  write: (Data) -> Bool, clientGone: () -> Bool) {
        let req: CompletionRequest
        do {
            req = try OpenAI.decode(r.body, kind: kind, defaults: config.defaults)
        } catch let e as OpenAIError {
            _ = write(errorResponse(e, origin: origin))
            return
        } catch {
            _ = write(errorResponse(OpenAIError(status: 400, message: "\(error)"), origin: origin))
            return
        }

        // Admission: the running request plus at most maxQueue waiting.
        queueLock.lock()
        if inFlight > config.maxQueue {
            queueLock.unlock()
            _ = write(errorResponse(OpenAIError(status: 503, message: "server busy, retry later", type: "server_error"),
                                    origin: origin))
            return
        }
        inFlight += 1
        queueLock.unlock()
        defer { queueLock.lock(); inFlight -= 1; queueLock.unlock() }

        let id = (kind == .chat ? "chatcmpl-" : "cmpl-") + String(UInt64.random(in: 0...UInt64.max), radix: 16)
        let created = Int(Date().timeIntervalSince1970)
        let model = backend.modelID

        if !req.stream {
            do {
                let result = try backend.run(req, shouldContinue: { !clientGone() }, onText: { _ in })
                let body = kind == .chat
                    ? OpenAI.chatResponse(id: id, model: model, created: created, result: result)
                    : OpenAI.textResponse(id: id, model: model, created: created, result: result)
                _ = write(jsonResponse(200, body, origin: origin))
            } catch {
                _ = write(errorResponse(OpenAIError(status: 500, message: "\(error)", type: "server_error"),
                                        origin: origin))
            }
            return
        }

        // Streaming: SSE head, a role chunk, one chunk per text piece, the final chunk.
        var alive = write(HTTP.sseHead(headers: corsHeaders(origin)))
        func event(_ o: Any) { if alive { alive = write(HTTP.sseEvent(JSON.string(o))) } }
        if kind == .chat { event(OpenAI.chunk(kind: kind, id: id, model: model, created: created, delta: nil, role: true)) }
        do {
            let result = try backend.run(req, shouldContinue: { alive }, onText: { text in
                event(OpenAI.chunk(kind: kind, id: id, model: model, created: created, delta: text))
            })
            event(OpenAI.chunk(kind: kind, id: id, model: model, created: created, delta: nil,
                               finish: OpenAI.finish(result)))
            if req.includeUsage {
                event(OpenAI.usageChunk(kind: kind, id: id, model: model, created: created, result: result))
            }
            event(["colibri": OpenAI.stats(result)])
        } catch {
            event(OpenAIError(status: 500, message: "\(error)", type: "server_error").body)
        }
        if alive { _ = write(HTTP.sseEvent("[DONE]")) }
    }
}
