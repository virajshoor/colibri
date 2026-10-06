// Operating-system glue: file I/O with pread, clocks, memory size, and the error type
// every layer of the engine throws.
//
// The engine reads weights with pread() into buffers it owns, never mmap. A mapped
// 400 GB model would count its touched pages against the process and let the kernel
// decide what stays resident; explicit reads keep resident memory equal to what the
// engine chose to cache.

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation

/// Error raised by the engine. The message is shown to the user as-is.
public struct ColibriError: Error, CustomStringConvertible, Sendable {
    public let description: String
    public init(_ message: String) { description = message }
}

/// A read-only file opened for positioned reads. Thread-safe: pread does not move a
/// shared file offset, so many threads may read the same file at once.
public final class ReadOnlyFile: @unchecked Sendable {
    public let path: String
    public let fd: Int32
    public let size: Int64

    public init(path: String) throws {
        // Work in locals: a class may not read its own properties before all are set.
        let f = open(path, O_RDONLY)
        guard f >= 0 else { throw ColibriError("cannot open \(path): \(String(cString: strerror(errno)))") }
        var st = stat()
        guard fstat(f, &st) == 0 else {
            close(f)
            throw ColibriError("cannot stat \(path)")
        }
        self.path = path
        fd = f
        size = Int64(st.st_size)
        #if canImport(Darwin)
        // F_NOCACHE: expert reads bypass the unified buffer cache. The engine keeps its
        // own expert cache; letting macOS also cache the same bytes would double the
        // memory each expert costs and evict the engine's working set.
        if ProcessInfo.processInfo.environment["COLI_PAGE_CACHE"] != "1" {
            _ = fcntl(f, F_NOCACHE, 1)
        }
        #endif
    }

    deinit { close(fd) }

    /// Reads exactly `count` bytes at `offset` into `buffer`. pread may return fewer
    /// bytes than asked (and macOS caps one call at INT_MAX), so it loops.
    public func read(into buffer: UnsafeMutableRawPointer, count: Int, offset: Int64) throws {
        var done = 0
        while done < count {
            let chunk = min(count - done, 1 << 30)
            let n = pread(fd, buffer + done, chunk, off_t(offset + Int64(done)))
            if n < 0 {
                if errno == EINTR { continue }
                throw ColibriError("read failed on \(path): \(String(cString: strerror(errno)))")
            }
            if n == 0 { throw ColibriError("unexpected end of file in \(path) at \(offset + Int64(done))") }
            done += n
        }
    }

    /// Reads `count` bytes at `offset` into a new array.
    public func readBytes(count: Int, offset: Int64) throws -> [UInt8] {
        var out = [UInt8](repeating: 0, count: count)
        try out.withUnsafeMutableBytes { try read(into: $0.baseAddress!, count: count, offset: offset) }
        return out
    }
}

/// Seconds from a monotonic clock, for timing.
@inline(__always) public func monotonicSeconds() -> Double {
    var ts = timespec()
    clock_gettime(CLOCK_MONOTONIC, &ts)
    return Double(ts.tv_sec) + Double(ts.tv_nsec) * 1e-9
}

/// Host memory, used to size the expert cache when no `--ram` budget is given.
public enum HostMemory {
    /// Physical RAM in bytes.
    public static var total: Int64 { Int64(ProcessInfo.processInfo.physicalMemory) }

    /// Memory the system could hand to this process now, in bytes. On macOS this is
    /// free + inactive + purgeable + speculative pages, which is what Activity Monitor
    /// reports as available; on Linux it is MemAvailable.
    public static var available: Int64 {
        #if canImport(Darwin)
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return total / 2 }
        let page = Int64(vm_kernel_page_size)
        let pages = Int64(stats.free_count) + Int64(stats.inactive_count)
            + Int64(stats.purgeable_count) + Int64(stats.speculative_count)
        return pages * page
        #else
        if let text = try? String(contentsOfFile: "/proc/meminfo", encoding: .utf8) {
            for line in text.split(separator: "\n") where line.hasPrefix("MemAvailable:") {
                let kb = line.split(separator: " ").compactMap { Int64($0) }.first ?? 0
                return kb * 1024
            }
        }
        return total / 2
        #endif
    }
}

/// Number of worker threads for parallel loops: the performance cores on Apple
/// Silicon when known, otherwise every active core. `COLI_THREADS` overrides.
public let workerThreadCount: Int = {
    if let s = ProcessInfo.processInfo.environment["COLI_THREADS"], let n = Int(s), n > 0 { return n }
    #if canImport(Darwin)
    var perf: Int32 = 0
    var size = MemoryLayout<Int32>.size
    if sysctlbyname("hw.perflevel0.physicalcpu", &perf, &size, nil, 0) == 0, perf > 0 { return Int(perf) }
    #endif
    return max(1, ProcessInfo.processInfo.activeProcessorCount)
}()

/// Runs `body(i)` for every i in 0..<n, split into contiguous chunks across worker
/// threads. Small loops run inline because waking threads costs more than the work.
public func parallelFor(_ n: Int, minPerThread: Int = 1, _ body: (Int) -> Void) {
    let threads = min(workerThreadCount, max(1, n / max(1, minPerThread)))
    if threads <= 1 {
        for i in 0..<n { body(i) }
        return
    }
    let chunk = (n + threads - 1) / threads
    DispatchQueue.concurrentPerform(iterations: threads) { t in
        let lo = t * chunk
        let hi = min(n, lo + chunk)
        if lo < hi { for i in lo..<hi { body(i) } }
    }
}

/// Bytes rendered for people, e.g. "12.4 GB".
public func formatBytes(_ b: Int64) -> String {
    let gb = Double(b) / 1e9
    if gb >= 1 { return String(format: "%.1f GB", gb) }
    return String(format: "%.0f MB", Double(b) / 1e6)
}
