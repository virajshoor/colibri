// colibrì's quantized weight formats and how a container's bytes identify them.
//
// A weight matrix W has O rows (outputs) and I columns (inputs). The container never
// stores a format number: the format follows from the byte count of W and of its
// `W.qs` scale array. This is a port of `qt_resolve_fmt` in the C engine and of the
// registry in docs/FORMATS.md; the ordinals match so logs and docs agree.
//
// | fmt | name          | weight bytes          | scales                          |
// |-----|---------------|-----------------------|---------------------------------|
// | 0   | f32           | O*I*4                 | none                            |
// | 1   | int8-row      | O*I                   | one per row                     |
// | 2   | int4-row      | O*ceil(I/2)           | one per row                     |
// | 3   | int2-row      | O*ceil(I/4)           | one per row                     |
// | 4   | int4-grouped  | O*ceil(I/2)           | one per group of gs inputs      |
// | 5   | int3-g64      | O*ceil(I/64)*24       | one per 64-input group          |
// | 6   | e8-iq3        | O*ceil(I/256)*98      | inside the blocks (not decoded) |
// | 8   | fp8-e4m3-b128 | O*I                   | one per 128x128 block           |

public enum QuantFormat: Equatable, Sendable, CustomStringConvertible {
    case f32
    case int8Row
    case int4Row
    case int2Row
    case int4Grouped(groupSize: Int)
    case int3G64
    case fp8Block128

    /// The ordinal used by the C engine and docs/FORMATS.md.
    public var ordinal: Int {
        switch self {
        case .f32: return 0
        case .int8Row: return 1
        case .int4Row: return 2
        case .int2Row: return 3
        case .int4Grouped: return 4
        case .int3G64: return 5
        case .fp8Block128: return 8
        }
    }

    public var description: String {
        switch self {
        case .f32: return "f32"
        case .int8Row: return "int8-row"
        case .int4Row: return "int4-row"
        case .int2Row: return "int2-row"
        case .int4Grouped(let g): return "int4-g\(g)"
        case .int3G64: return "int3-g64"
        case .fp8Block128: return "fp8-e4m3-b128"
        }
    }

    /// Format from a `colibri.fmt` metadata stamp name, or nil if unknown.
    static func fromStamp(_ name: String) -> Int? {
        switch name {
        case "f32": return 0
        case "int8-row": return 1
        case "int4-row": return 2
        case "int2-row": return 3
        case "int4-grouped": return 4
        case "int3-g64": return 5
        case "e8-iq3-lattice": return 6
        case "fp8-e4m3-b128": return 8
        default: return nil
        }
    }

    // Byte geometry shared with the kernels.
    static func int3Groups(_ i: Int) -> Int { (i + 63) / 64 }
    static func int3RowBytes(_ i: Int) -> Int { int3Groups(i) * 24 }
    static func e8RowBytes(_ i: Int) -> Int { (i + 255) / 256 * 98 }
    static func fp8Blocks(_ n: Int) -> Int { (n + 127) / 128 }

    /// Finest group size whose scale count matches `scaleBytes`, or 0 if the scales
    /// are per row. Same candidates, in the same order, as the C engine.
    static func detectGroupSize(o: Int, i: Int, scaleBytes: Int) -> Int {
        guard o > 0, i > 0, scaleBytes > o * 4 else { return 0 }
        for gs in [16, 32, 48, 64, 96, 128, 192, 256] {
            if gs > i { break }
            if scaleBytes == o * ((i + gs - 1) / gs) * 4 { return gs }
        }
        return 0
    }

    /// Identifies the format of an [o, i] weight stored in `weightBytes` bytes with
    /// `scaleBytes` bytes of scales. Throws on any layout that does not fit exactly,
    /// because the container is untrusted input.
    public static func resolve(name: String, o: Int, i: Int, weightBytes nb: Int, scaleBytes ns: Int,
                               stamp: String?) throws -> QuantFormat {
        let stamped = stamp.flatMap(fromStamp)
        // E8/IQ3 lattice (fmt 6) is recognised so the error names it, but this build
        // has no decoder for it.
        if ns == 4 && nb == o * e8RowBytes(i) && !(nb == o * i && (o == 1 || stamped == 8 || stamped == 1)) {
            throw ColibriError("\(name): E8/IQ3 lattice weights (fmt 6) are not supported by the Swift engine; reconvert with int4/int3")
        }
        var fmt: Int
        if nb == o * i { fmt = 1 }
        else if nb == o * ((i + 1) / 2) { fmt = 2 }
        else if nb == o * ((i + 3) / 4) { fmt = 3 }
        else if nb == o * int3RowBytes(i) { fmt = 5 }
        else {
            throw ColibriError("\(name): \(nb) weight bytes match no int8/int4/int2/int3/fp8 layout for [\(o),\(i)]")
        }
        var groupSize = 0
        if fmt == 2 {
            let g = detectGroupSize(o: o, i: i, scaleBytes: ns)
            if g > 0 { fmt = 4; groupSize = g }
        }
        if fmt == 1 {
            // int8 per-row and FP8 per-block have identical weight bytes; the scale
            // array's size tells them apart, and a stamp settles the rare shapes where
            // both sizes coincide.
            let blocks = fp8Blocks(o) * fp8Blocks(i)
            let isRow = ns == o * 4, isBlock = ns == blocks * 4
            if isRow && isBlock {
                if stamped == 8 { fmt = 8 }  // otherwise int8-row, the incumbent
            } else if isBlock {
                fmt = 8
            } else if ns == blocks && !isRow {
                throw ColibriError("\(name): FP8 with UE8M0 block scales is recognised but not implemented")
            }
        }
        let expectScales: Int
        switch fmt {
        case 4: expectScales = o * ((i + groupSize - 1) / groupSize)
        case 5: expectScales = o * int3Groups(i)
        case 8: expectScales = fp8Blocks(o) * fp8Blocks(i)
        default: expectScales = o
        }
        guard ns == expectScales * 4 else {
            throw ColibriError("\(name): scale array is \(ns) bytes, expected \(expectScales * 4) for [\(o),\(i)] fmt \(fmt)")
        }
        switch fmt {
        case 1: return .int8Row
        case 2: return .int4Row
        case 3: return .int2Row
        case 4: return .int4Grouped(groupSize: groupSize)
        case 5: return .int3G64
        default: return .fp8Block128
        }
    }
}
