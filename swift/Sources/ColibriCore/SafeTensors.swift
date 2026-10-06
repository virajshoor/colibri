// Index and read tensors spread across the safetensors shards of a model directory.
//
// A safetensors file is: an 8-byte little-endian header length, a JSON header that maps
// each tensor name to {dtype, shape, data_offsets}, then the raw tensor bytes. Opening a
// shard reads only the header; tensor bytes are read later, on demand, with pread.
// colibrì's converted containers store each quantized weight `W` as two tensors: `W`
// (packed integer bytes) and `W.qs` (its float scales).

import Foundation

/// Element type of a stored tensor.
public enum DType: String, Sendable {
    case bf16 = "BF16", f16 = "F16", f32 = "F32", u8 = "U8", i8 = "I8"
    case f8e4m3 = "F8_E4M3", f8e8m0 = "F8_E8M0", i64 = "I64", u64 = "U64", i32 = "I32"

    /// Accepts the spellings different exporters use for the same type.
    init?(header s: String) {
        switch s {
        case "F8_E4M3FN", "float8_e4m3fn": self = .f8e4m3
        case "F8_E8M0FNU": self = .f8e8m0
        default: self.init(rawValue: s)
        }
    }

    /// Bytes per element.
    public var size: Int {
        switch self {
        case .f32, .i32: return 4
        case .u8, .i8, .f8e4m3, .f8e8m0: return 1
        case .i64, .u64: return 8
        case .bf16, .f16: return 2
        }
    }
}

/// Where one tensor's bytes live.
public struct TensorInfo: Sendable {
    public let name: String
    public let file: ReadOnlyFile
    public let offset: Int64  // absolute offset of the first data byte in `file`
    public let byteCount: Int
    public let dtype: DType
    public let shape: [Int]
    public var count: Int { shape.reduce(1, *) }
}

/// Every tensor of a model directory, looked up by name.
public final class ShardSet: @unchecked Sendable {
    public let directory: URL
    public private(set) var tensors: [String: TensorInfo] = [:]
    /// Format names from the optional `colibri.fmt` metadata stamp, by tensor name.
    public private(set) var formatStamps: [String: String] = [:]
    public private(set) var files: [ReadOnlyFile] = []

    /// Headers larger than this are refused: real ones are kilobytes to a few
    /// megabytes, and an untrusted file must not make us allocate gigabytes.
    static let maxHeader: Int64 = 512 << 20

    public init(directory: URL) throws {
        self.directory = directory
        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
            .filter { $0.hasSuffix(".safetensors") }
            .sorted()
        guard !names.isEmpty else { throw ColibriError("no .safetensors files in \(directory.path)") }
        for n in names { try addShard(directory.appendingPathComponent(n).path) }
    }

    /// Parses one shard's header and registers its tensors.
    private func addShard(_ path: String) throws {
        let f = try ReadOnlyFile(path: path)
        files.append(f)
        let lenBytes = try f.readBytes(count: 8, offset: 0)
        var hlen: Int64 = 0
        for i in (0..<8).reversed() { hlen = hlen << 8 | Int64(lenBytes[i]) }
        guard hlen > 0, hlen <= Self.maxHeader, 8 + hlen <= f.size else {
            throw ColibriError("\(path): bad safetensors header length \(hlen)")
        }
        let header = try f.readBytes(count: Int(hlen), offset: 8)
        guard let root = try JSONSerialization.jsonObject(with: Data(header)) as? [String: Any] else {
            throw ColibriError("\(path): header is not a JSON object")
        }
        let dataStart = 8 + hlen
        for (name, value) in root {
            if name == "__metadata__" {
                if let meta = value as? [String: Any], let stamp = meta["colibri.fmt"] as? String {
                    parseStamp(stamp)
                }
                continue
            }
            guard let d = value as? [String: Any],
                  let ds = d["dtype"] as? String,
                  let shape = d["shape"] as? [Int],
                  let offs = d["data_offsets"] as? [Int], offs.count == 2
            else { throw ColibriError("\(path): malformed entry for \(name)") }
            guard let dtype = DType(header: ds) else { throw ColibriError("\(path): unsupported dtype \(ds) for \(name)") }
            let start = Int64(offs[0]), end = Int64(offs[1])
            let count = shape.reduce(1, *)
            // Reject entries whose byte range is inconsistent or runs past the file.
            guard start >= 0, end >= start, dataStart + end <= f.size,
                  Int(end - start) == count * dtype.size
            else { throw ColibriError("\(path): \(name) has an invalid byte range") }
            tensors[name] = TensorInfo(
                name: name, file: f, offset: dataStart + start, byteCount: Int(end - start),
                dtype: dtype, shape: shape)
        }
    }

    /// The stamp is JSON text inside a metadata string: {"tensor name": "format name"}.
    private func parseStamp(_ text: String) {
        guard let data = text.data(using: .utf8),
              let map = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return }
        for (k, v) in map { if let s = v as? String { formatStamps[k] = s } }
    }

    public func has(_ name: String) -> Bool { tensors[name] != nil }

    public func info(_ name: String) throws -> TensorInfo {
        guard let t = tensors[name] else { throw ColibriError("tensor not found: \(name)") }
        return t
    }

    /// Reads a tensor's raw bytes into caller-owned memory of at least `byteCount` bytes.
    public func readRaw(_ name: String, into dst: UnsafeMutableRawPointer) throws {
        let t = try info(name)
        try t.file.read(into: dst, count: t.byteCount, offset: t.offset)
    }

    /// Reads a tensor and converts it to Float. Accepts BF16, F16, F32, FP8 E4M3 and
    /// E8M0; integer tensors are refused because their meaning depends on a scale.
    public func readFloats(_ name: String, expectCount: Int? = nil) throws -> [Float] {
        let t = try info(name)
        if let n = expectCount, n != t.count {
            throw ColibriError("\(name): expected \(n) values, file has \(t.count) \(t.shape)")
        }
        let raw = try t.file.readBytes(count: t.byteCount, offset: t.offset)
        var out = [Float](repeating: 0, count: t.count)
        raw.withUnsafeBytes { src in
            out.withUnsafeMutableBufferPointer { dst in
                switch t.dtype {
                case .f32:
                    for i in 0..<t.count { dst[i] = src.loadUnaligned(fromByteOffset: i * 4, as: Float.self) }
                case .bf16:
                    for i in 0..<t.count { dst[i] = bf16ToFloat(src.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)) }
                case .f16:
                    for i in 0..<t.count { dst[i] = f16ToFloat(src.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)) }
                case .f8e4m3:
                    for i in 0..<t.count { dst[i] = e4m3Table[Int(src[i])] }
                case .f8e8m0:
                    for i in 0..<t.count { dst[i] = e8m0ToFloat(src[i]) }
                default: break
                }
            }
        }
        switch t.dtype {
        case .f32, .bf16, .f16, .f8e4m3, .f8e8m0: return out
        default: throw ColibriError("\(name): \(t.dtype.rawValue) cannot be read as float")
        }
    }
}
