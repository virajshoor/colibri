// Minimal HTTP/1.1: parse one request from bytes, serialise a response. Pure functions
// with no sockets, so they are unit-tested directly. Each connection carries one
// request and is closed afterwards (`Connection: close`), which keeps streaming and
// cancellation simple and is what OpenAI clients handle fine.

import Foundation

public struct HTTPRequest: Sendable, Equatable {
    public var method: String
    public var path: String          // without the query string, percent-decoded
    public var query: String
    public var headers: [String: String]  // lower-cased names
    public var body: Data

    public func header(_ name: String) -> String? { headers[name.lowercased()] }
}

public enum HTTPParseResult: Equatable {
    case incomplete
    case complete(HTTPRequest)
    case failure(status: Int, message: String)
}

public enum HTTP {
    public static let maxHeaderBytes = 64 * 1024

    /// Parses a request from the bytes received so far.
    public static func parse(_ buf: [UInt8], maxBody: Int) -> HTTPParseResult {
        // Find the blank line ending the head.
        let crlf2: [UInt8] = [13, 10, 13, 10]
        var headEnd = -1
        if buf.count >= 4 {
            for i in 0...(buf.count - 4) where buf[i] == 13 && Array(buf[i..<i + 4]) == crlf2 {
                headEnd = i
                break
            }
        }
        if headEnd < 0 {
            return buf.count > maxHeaderBytes ? .failure(status: 431, message: "request header too large") : .incomplete
        }
        guard let head = String(bytes: buf[0..<headEnd], encoding: .utf8) else {
            return .failure(status: 400, message: "header is not UTF-8")
        }
        var lines = head.components(separatedBy: "\r\n")
        let requestLine = lines.removeFirst().split(separator: " ", omittingEmptySubsequences: true)
        guard requestLine.count == 3, requestLine[2].hasPrefix("HTTP/1.") else {
            return .failure(status: 400, message: "bad request line")
        }
        var headers: [String: String] = [:]
        for line in lines where !line.isEmpty {
            guard let colon = line.firstIndex(of: ":") else { return .failure(status: 400, message: "bad header line") }
            let name = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            headers[name] = value
        }
        if let te = headers["transfer-encoding"], te.lowercased() != "identity" {
            return .failure(status: 411, message: "chunked request bodies are not supported; send Content-Length")
        }
        var length = 0
        if let cl = headers["content-length"] {
            guard let n = Int(cl), n >= 0 else { return .failure(status: 400, message: "bad Content-Length") }
            length = n
        }
        if length > maxBody { return .failure(status: 413, message: "request body too large") }
        let bodyStart = headEnd + 4
        if buf.count - bodyStart < length { return .incomplete }

        let target = String(requestLine[1])
        let parts = target.split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false)
        let rawPath = String(parts[0])
        let path = rawPath.removingPercentEncoding ?? rawPath
        return .complete(HTTPRequest(
            method: String(requestLine[0]).uppercased(), path: path,
            query: parts.count > 1 ? String(parts[1]) : "",
            headers: headers, body: Data(buf[bodyStart..<bodyStart + length])))
    }

    public static func reason(_ status: Int) -> String {
        switch status {
        case 200: return "OK"
        case 204: return "No Content"
        case 400: return "Bad Request"
        case 401: return "Unauthorized"
        case 403: return "Forbidden"
        case 404: return "Not Found"
        case 405: return "Method Not Allowed"
        case 408: return "Request Timeout"
        case 411: return "Length Required"
        case 413: return "Payload Too Large"
        case 421: return "Misdirected Request"
        case 431: return "Request Header Fields Too Large"
        case 500: return "Internal Server Error"
        case 503: return "Service Unavailable"
        default: return "Status"
        }
    }

    /// Serialises a complete response with a body.
    public static func response(status: Int, headers: [(String, String)] = [], body: Data) -> Data {
        var head = "HTTP/1.1 \(status) \(reason(status))\r\n"
        for (k, v) in headers { head += "\(k): \(v)\r\n" }
        head += "Content-Length: \(body.count)\r\nConnection: close\r\n\r\n"
        var out = Data(head.utf8)
        out.append(body)
        return out
    }

    /// Head of a Server-Sent Events stream; events follow with `sseEvent`.
    public static func sseHead(headers: [(String, String)] = []) -> Data {
        var head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n"
        for (k, v) in headers { head += "\(k): \(v)\r\n" }
        return Data((head + "\r\n").utf8)
    }

    /// One SSE event carrying `data`.
    public static func sseEvent(_ data: String) -> Data { Data("data: \(data)\n\n".utf8) }
}

/// JSON helpers on Foundation's serializer.
public enum JSON {
    public static func encode(_ object: Any) -> Data {
        (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes]))
            ?? Data("{}".utf8)
    }

    public static func string(_ object: Any) -> String { String(decoding: encode(object), as: UTF8.self) }

    public static func decodeObject(_ data: Data) throws -> [String: Any] {
        guard let o = try? JSONSerialization.jsonObject(with: data), let d = o as? [String: Any] else {
            throw OpenAIError(status: 400, message: "request body must be a JSON object")
        }
        return d
    }
}
