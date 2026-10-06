// Small vector kernels used by the forward pass. On Apple platforms the dot products
// and AXPYs go through Accelerate (vDSP), which uses the NEON units and, for larger
// vectors, the AMX coprocessor; elsewhere they are plain loops the compiler vectorises.

#if canImport(Accelerate)
import Accelerate
#endif
import Foundation

public typealias FloatPtr = UnsafeMutablePointer<Float>
public typealias ConstFloatPtr = UnsafePointer<Float>

/// Sum of a[i]*b[i] for i < n.
@inline(__always) public func dot(_ a: ConstFloatPtr, _ b: ConstFloatPtr, _ n: Int) -> Float {
    #if canImport(Accelerate)
    var r: Float = 0
    vDSP_dotpr(a, 1, b, 1, &r, vDSP_Length(n))
    return r
    #else
    var r: Float = 0
    for i in 0..<n { r += a[i] * b[i] }
    return r
    #endif
}

/// y[i] += alpha * x[i] for i < n.
@inline(__always) public func axpy(_ alpha: Float, _ x: ConstFloatPtr, _ y: FloatPtr, _ n: Int) {
    #if canImport(Accelerate)
    var a = alpha
    vDSP_vsma(x, 1, &a, y, 1, y, 1, vDSP_Length(n))
    #else
    for i in 0..<n { y[i] += alpha * x[i] }
    #endif
}

/// out = x / rms(x) * w. Accumulates in Double like the C engine so long rows do not
/// lose precision.
public func rmsNorm(_ out: FloatPtr, _ x: ConstFloatPtr, _ w: ConstFloatPtr, _ n: Int, eps: Float) {
    var ms: Double = 0
    for i in 0..<n { ms += Double(x[i]) * Double(x[i]) }
    let r = 1 / (Float(ms / Double(n)) + eps).squareRoot()
    for i in 0..<n { out[i] = x[i] * r * w[i] }
}

/// In-place softmax over n values.
public func softmax(_ x: FloatPtr, _ n: Int) {
    var m = -Float.infinity
    for i in 0..<n where x[i] > m { m = x[i] }
    var s: Float = 0
    for i in 0..<n { x[i] = expf(x[i] - m); s += x[i] }
    let inv = 1 / s
    for i in 0..<n { x[i] *= inv }
}

@inline(__always) public func sigmoid(_ x: Float) -> Float { 1 / (1 + expf(-x)) }
@inline(__always) public func silu(_ x: Float) -> Float { x / (1 + expf(-x)) }

/// Rotary position embedding in GLM/DeepSeek's layout: the input pairs are interleaved
/// (x0,x1),(x2,x3)... and the output is split in halves, first the rotated even
/// elements, then the rotated odd ones. Queries and keys both go through this, so their
/// dot products see the same rotation.
public func ropeInterleave(_ v: FloatPtr, dim: Int, pos: Int, theta: Float) {
    let half = dim / 2
    var tmp = [Float](repeating: 0, count: dim)
    for i in 0..<dim { tmp[i] = v[i] }
    for j in 0..<half {
        let inv = powf(theta, -2 * Float(j) / Float(dim))
        let ang = Float(pos) * inv
        let c = cosf(ang), s = sinf(ang)
        let a = tmp[2 * j], b = tmp[2 * j + 1]
        v[j] = a * c - b * s
        v[half + j] = b * c + a * s
    }
}

/// Splits 0..<n into one contiguous range per worker thread and runs `body` on each.
/// Each call gets a range, so it can allocate scratch memory once per thread.
public func parallelRanges(_ n: Int, minPerThread: Int = 1, _ body: (Range<Int>) -> Void) {
    let threads = min(workerThreadCount, max(1, n / max(1, minPerThread)))
    if threads <= 1 {
        if n > 0 { body(0..<n) }
        return
    }
    let chunk = (n + threads - 1) / threads
    DispatchQueue.concurrentPerform(iterations: threads) { t in
        let lo = t * chunk, hi = min(n, lo + chunk)
        if lo < hi { body(lo..<hi) }
    }
}
