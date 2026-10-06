// A weight matrix [O, I] held in one of colibrì's quantized formats, with the kernels
// that use it: dequantise a row, y = x·Wᵀ for a batch of rows, and the transposed
// accumulation the attention needs.
//
// Kernels decode one row into a small Float buffer (it stays in L1: I is at most a few
// thousand) and then reuse it for every input row of the batch, so a prefill of S
// tokens decodes each weight once rather than S times. Rows are split across threads.

import Foundation

public final class QuantTensor: @unchecked Sendable {
    public let format: QuantFormat
    public let rows: Int  // O
    public let cols: Int  // I
    /// Packed weights (or Float values for `.f32`). Owned; freed in deinit.
    let weights: UnsafeMutableRawPointer
    public let weightBytes: Int
    /// Scales: per row, per group, or per 128x128 block depending on the format.
    let scales: [Float]

    /// Bytes this tensor keeps in memory.
    public var residentBytes: Int { weightBytes + scales.count * 4 }

    init(format: QuantFormat, rows: Int, cols: Int, weightBytes: Int, scales: [Float]) {
        self.format = format
        self.rows = rows
        self.cols = cols
        self.weightBytes = weightBytes
        self.weights = UnsafeMutableRawPointer.allocate(byteCount: max(weightBytes, 1), alignment: 64)
        self.scales = scales
    }

    deinit { weights.deallocate() }

    /// An f32 tensor built from values in memory (tests and the router).
    public convenience init(rows: Int, cols: Int, values: [Float]) {
        precondition(values.count == rows * cols)
        self.init(format: .f32, rows: rows, cols: cols, weightBytes: rows * cols * 4, scales: [])
        values.withUnsafeBytes { weights.copyMemory(from: $0.baseAddress!, byteCount: rows * cols * 4) }
    }

    /// A quantized tensor built from packed bytes and scales in memory (tests).
    public convenience init(format: QuantFormat, rows: Int, cols: Int, packed: [UInt8], scales: [Float]) {
        self.init(format: format, rows: rows, cols: cols, weightBytes: packed.count, scales: scales)
        packed.withUnsafeBytes { weights.copyMemory(from: $0.baseAddress!, byteCount: packed.count) }
    }

    /// Loads `name` as an [o, i] matrix. A quantized weight has a `name.qs` scale
    /// tensor beside it; without one the tensor is read as plain floats.
    public static func load(_ shards: ShardSet, _ name: String, rows o: Int, cols i: Int) throws -> QuantTensor {
        let scaleName = name + ".qs"
        if shards.has(scaleName) {
            let w = try shards.info(name)
            let s = try shards.info(scaleName)
            let fmt = try QuantFormat.resolve(
                name: name, o: o, i: i, weightBytes: w.byteCount, scaleBytes: s.count * 4,
                stamp: shards.formatStamps[name])
            let scales = try shards.readFloats(scaleName)
            let t = QuantTensor(format: fmt, rows: o, cols: i, weightBytes: w.byteCount, scales: scales)
            try shards.readRaw(name, into: t.weights)
            return t
        }
        let values = try shards.readFloats(name, expectCount: o * i)
        return QuantTensor(rows: o, cols: i, values: values)
    }

    /// Decodes row `o` into `dst[0..<cols]`.
    public func dequantRow(_ o: Int, into dst: FloatPtr) {
        let n = cols
        switch format {
        case .f32:
            let src = weights.assumingMemoryBound(to: Float.self) + o * n
            dst.update(from: src, count: n)
        case .int8Row:
            let w = weights.assumingMemoryBound(to: Int8.self) + o * n
            let s = scales[o]
            for i in 0..<n { dst[i] = Float(w[i]) * s }
        case .int4Row:
            // Two values per byte, low nibble first, stored as v+8.
            let w = weights.assumingMemoryBound(to: UInt8.self) + o * ((n + 1) / 2)
            let s = scales[o]
            var i = 0
            while i + 1 < n {
                let b = w[i >> 1]
                dst[i] = Float(Int(b & 0xF) - 8) * s
                dst[i + 1] = Float(Int(b >> 4) - 8) * s
                i += 2
            }
            if i < n { dst[i] = Float(Int(w[i >> 1] & 0xF) - 8) * s }
        case .int4Grouped(let gs):
            let w = weights.assumingMemoryBound(to: UInt8.self) + o * ((n + 1) / 2)
            let ng = (n + gs - 1) / gs
            scales.withUnsafeBufferPointer { sc in
                let srow = sc.baseAddress! + o * ng
                for i in 0..<n {
                    let b = w[i >> 1]
                    let q = (i & 1) == 0 ? Int(b & 0xF) : Int(b >> 4)
                    dst[i] = Float(q - 8) * srow[i / gs]
                }
            }
        case .int2Row:
            // Four values per byte, lowest bits first, stored as v+2.
            let w = weights.assumingMemoryBound(to: UInt8.self) + o * ((n + 3) / 4)
            let s = scales[o]
            for i in 0..<n { dst[i] = Float(Int((w[i >> 2] >> UInt8((i & 3) * 2)) & 3) - 2) * s }
        case .int3G64:
            // Per 64-value group: 16 bytes of low 2-bit planes, then 8 bytes holding the
            // third bit of each value; values are stored as v+4.
            let ng = QuantFormat.int3Groups(n)
            let w = weights.assumingMemoryBound(to: UInt8.self) + o * ng * 24
            scales.withUnsafeBufferPointer { sc in
                for g in 0..<ng {
                    let lo = w + g * 24, hi = lo + 16
                    let s = sc[o * ng + g]
                    let base = g * 64
                    for k in 0..<min(64, n - base) {
                        let low = (lo[k >> 2] >> UInt8((k & 3) * 2)) & 3
                        let high = (hi[k >> 3] >> UInt8(k & 7)) & 1
                        dst[base + k] = Float(Int(low | (high << 2)) - 4) * s
                    }
                }
            }
        case .fp8Block128:
            let w = weights.assumingMemoryBound(to: UInt8.self) + o * n
            let nbI = QuantFormat.fp8Blocks(n)
            let blockRow = (o / 128) * nbI
            e4m3Table.withUnsafeBufferPointer { lut in
                scales.withUnsafeBufferPointer { sc in
                    for i in 0..<n { dst[i] = lut[Int(w[i])] * sc[blockRow + i / 128] }
                }
            }
        }
    }

    /// y[s, o] = Σ_i x[s, i] · W[o, i] for s < count. `x` is [count, cols] and `y` is
    /// [count, rows], both row-major.
    public func matmul(_ x: ConstFloatPtr, count: Int, into y: FloatPtr) {
        matmulRows(0..<rows, x, count: count, into: y, yStride: rows, yOffset: 0)
    }

    /// Like `matmul` but only for the output rows in `range`; result row o lands at
    /// y[s*yStride + yOffset + (o - range.lowerBound)].
    public func matmulRows(_ range: Range<Int>, _ x: ConstFloatPtr, count: Int, into y: FloatPtr,
                           yStride: Int, yOffset: Int, parallel: Bool = true) {
        let n = cols
        let lo = range.lowerBound
        let work: (Range<Int>) -> Void = { sub in
            let tmp = FloatPtr.allocate(capacity: n)
            defer { tmp.deallocate() }
            for o in sub {
                let r = lo + o
                self.dequantRow(r, into: tmp)
                for s in 0..<count {
                    y[s * yStride + yOffset + o] = dot(tmp, x + s * n, n)
                }
            }
        }
        if parallel { parallelRanges(range.count, minPerThread: 16, work) } else { work(0..<range.count) }
    }

    /// acc[0..<cols] += Σ_{o in range} coeff[o - range.lowerBound] · W[o, :]
    /// (x·W restricted to some rows: used to fold a query into the attention latent).
    public func accumulateTransposed(_ range: Range<Int>, coeff: ConstFloatPtr, into acc: FloatPtr) {
        let n = cols
        let tmp = FloatPtr.allocate(capacity: n)
        defer { tmp.deallocate() }
        for (k, o) in range.enumerated() {
            dequantRow(o, into: tmp)
            axpy(coeff[k], tmp, acc, n)
        }
    }
}
