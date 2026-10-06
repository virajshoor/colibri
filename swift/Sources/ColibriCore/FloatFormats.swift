// Conversions from the small float encodings found in checkpoints to Float.
//
// BF16 is the top half of an IEEE float, so widening is a shift. F16 has a different
// exponent width and goes through Float16 on Apple Silicon (a hardware conversion) or
// the bit algebra below elsewhere. FP8 E4M3 and E8M0 are the encodings DeepSeek and
// GLM ship their native FP8 weights and block scales in.

@inline(__always) public func bf16ToFloat(_ h: UInt16) -> Float {
    Float(bitPattern: UInt32(h) << 16)
}

@inline(__always) public func f16ToFloat(_ h: UInt16) -> Float {
    #if arch(arm64)
    return Float(Float16(bitPattern: h))
    #else
    let sign = UInt32(h & 0x8000) << 16
    var exp = UInt32((h >> 10) & 0x1F)
    var man = UInt32(h & 0x3FF)
    if exp == 0 {
        if man == 0 { return Float(bitPattern: sign) }  // signed zero
        // Subnormal: shift the mantissa up until its leading 1 is in the implicit bit.
        exp = 127 - 15 + 1
        while man & 0x400 == 0 { man <<= 1; exp -= 1 }
        man &= 0x3FF
        return Float(bitPattern: sign | (exp << 23) | (man << 13))
    }
    if exp == 0x1F { return Float(bitPattern: sign | 0x7F80_0000 | (man << 13)) }  // inf / NaN
    return Float(bitPattern: sign | ((exp - 15 + 127) << 23) | (man << 13))
    #endif
}

/// FP8 E4M3 ("fn" variant: no infinities, 0x7F/0xFF are NaN), exponent bias 7.
@inline(__always) public func e4m3ToFloat(_ b: UInt8) -> Float {
    let sign: Float = (b & 0x80) != 0 ? -1 : 1
    let exp = Int((b >> 3) & 0x0F)
    let man = Float(b & 0x07)
    if exp == 0x0F && (b & 0x07) == 0x07 { return .nan }
    if exp == 0 { return sign * man / 8 * 0.015625 }  // subnormal: man/8 * 2^-6
    return sign * (1 + man / 8) * Float(sign: .plus, exponent: exp - 7, significand: 1)
}

/// UE8M0: an unsigned power-of-two exponent with bias 127 (0xFF is NaN), used for
/// per-block scales in native FP8 checkpoints.
@inline(__always) public func e8m0ToFloat(_ b: UInt8) -> Float {
    if b == 0xFF { return .nan }
    return Float(sign: .plus, exponent: Int(b) - 127, significand: 1)
}

/// Lookup table for E4M3 so the hot loop is one load per weight.
public let e4m3Table: [Float] = (0..<256).map { e4m3ToFloat(UInt8($0)) }
