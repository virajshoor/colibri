// Tests for the Swift engine: number formats, quantized kernels, the container
// reader, the usage history, sampling, argument parsing, and a full forward pass of a
// tiny model whose experts stream from disk.

import Foundation
import Testing
@testable import ColibriCore

// MARK: float encodings

@Test func floatEncodings() {
    #expect(bf16ToFloat(0x3F80) == 1)
    #expect(bf16ToFloat(0xC000) == -2)
    #expect(f16ToFloat(0x3C00) == 1)
    #expect(f16ToFloat(0xC000) == -2)
    #expect(f16ToFloat(0x0001) == Float(5.9604645e-8))  // smallest subnormal
    #expect(e4m3ToFloat(0x38) == 1)      // exponent 7 = bias
    #expect(e4m3ToFloat(0xC0) == -2)
    #expect(e4m3ToFloat(0x7E) == 448)    // largest finite
    #expect(e4m3ToFloat(0x7F).isNaN)
    #expect(e8m0ToFloat(127) == 1)
    #expect(e8m0ToFloat(130) == 8)
}

// MARK: format resolution

@Test func resolvesFormatsFromByteCounts() throws {
    func r(_ o: Int, _ i: Int, _ nb: Int, _ ns: Int) throws -> QuantFormat {
        try QuantFormat.resolve(name: "w", o: o, i: i, weightBytes: nb, scaleBytes: ns, stamp: nil)
    }
    #expect(try r(4, 300, 4 * 300, 4 * 4) == .int8Row)
    #expect(try r(4, 300, 4 * 150, 4 * 4) == .int4Row)
    #expect(try r(4, 300, 4 * 75, 4 * 4) == .int2Row)
    #expect(try r(4, 300, 4 * 5 * 24, 4 * 5 * 4) == .int3G64)
    #expect(try r(4, 256, 4 * 128, 4 * 4 * 4) == .int4Grouped(groupSize: 64))
    #expect(try r(256, 300, 256 * 300, 2 * 3 * 4) == .fp8Block128)
    #expect(throws: ColibriError.self) { try r(4, 300, 123, 16) }          // no layout
    #expect(throws: ColibriError.self) { try r(4, 300, 4 * 150, 12) }      // wrong scale count
}

// MARK: kernels

/// Dequantised rows must match the format definitions in docs/FORMATS.md.
@Test func dequantizesEveryFormat() {
    let out = FloatPtr.allocate(capacity: 128)
    defer { out.deallocate() }

    // int8: value * row scale.
    var t = QuantTensor(format: .int8Row, rows: 1, cols: 3, packed: [1, 0xFF, 0x80], scales: [0.5])
    t.dequantRow(0, into: out)
    #expect([out[0], out[1], out[2]] == [0.5, -0.5, -64])

    // int4: low nibble first, value + 8.
    t = QuantTensor(format: .int4Row, rows: 1, cols: 3, packed: [0x9F, 0x00], scales: [2])
    t.dequantRow(0, into: out)
    #expect([out[0], out[1], out[2]] == [14, 2, -16])

    // int2: four per byte, lowest bits first, value + 2.
    t = QuantTensor(format: .int2Row, rows: 1, cols: 4, packed: [0b11_10_01_00], scales: [1])
    t.dequantRow(0, into: out)
    #expect([out[0], out[1], out[2], out[3]] == [-2, -1, 0, 1])

    // int4 grouped by 2: one scale per pair.
    t = QuantTensor(format: .int4Grouped(groupSize: 2), rows: 1, cols: 4, packed: [0x99, 0x99], scales: [1, 3])
    t.dequantRow(0, into: out)
    #expect([out[0], out[1], out[2], out[3]] == [1, 1, 3, 3])

    // int3-g64: value k = low 2 bits from the low plane | high bit << 2, minus 4.
    var g = [UInt8](repeating: 0, count: 24)
    g[0] = 0b00_00_11_01          // k0 low=1, k1 low=3
    g[16] = 0b10                   // k1 high bit
    t = QuantTensor(format: .int3G64, rows: 1, cols: 64, packed: g, scales: [0.5])
    t.dequantRow(0, into: out)
    #expect(out[0] == (1 - 4) * 0.5)
    #expect(out[1] == (7 - 4) * 0.5)
    #expect(out[2] == -2)

    // fp8 e4m3 with one block scale.
    t = QuantTensor(format: .fp8Block128, rows: 1, cols: 2, packed: [0x38, 0xC0], scales: [3])
    t.dequantRow(0, into: out)
    #expect([out[0], out[1]] == [3, -6])
}

/// matmul over a batch equals dequantise-then-dot, for a quantized matrix.
@Test func matmulMatchesReference() {
    var rng = TestRNG()
    let o = 37, i = 70, s = 3
    let w = rng.values(o * i)
    let (p, sc) = quantizeInt4Row(w, rows: o, cols: i)
    let t = QuantTensor(format: .int4Row, rows: o, cols: i, packed: p, scales: sc)
    let x = rng.values(s * i)
    var y = [Float](repeating: 0, count: s * o)
    x.withUnsafeBufferPointer { xp in y.withUnsafeMutableBufferPointer { t.matmul(xp.baseAddress!, count: s, into: $0.baseAddress!) } }
    let row = FloatPtr.allocate(capacity: i)
    defer { row.deallocate() }
    for oo in 0..<o {
        t.dequantRow(oo, into: row)
        for ss in 0..<s {
            var ref: Float = 0
            for ii in 0..<i { ref += row[ii] * x[ss * i + ii] }
            #expect(abs(ref - y[ss * o + oo]) < 1e-4)
        }
    }
}

// MARK: containers

@Test func readsSafetensorsAndStamps() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("st-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: url) }
    let bf: [UInt16] = [0x3F80, 0x4000]
    try writeSafetensors(
        ["a": .f32([2, 2], [1, 2, 3, 4]),
         "b": RawTensor(dtype: "BF16", shape: [2], bytes: bf.withUnsafeBytes { Array($0) })],
        to: url.appendingPathComponent("x.safetensors"),
        metadata: ["colibri.fmt": "{\"a\": \"int8-row\"}"])
    let s = try ShardSet(directory: url)
    #expect(try s.readFloats("a") == [1, 2, 3, 4])
    #expect(try s.readFloats("b") == [1, 2])
    #expect(s.formatStamps["a"] == "int8-row")
    #expect(throws: ColibriError.self) { try s.readFloats("a", expectCount: 5) }
}

// MARK: usage history

@Test func usageHistoryRoundTrip() throws {
    var c = Array(repeating: [UInt32](repeating: 0, count: 4), count: 3)
    c[1][2] = 7
    c[2][0] = 3
    let text = UsageHistory.render(c)
    #expect(text.hasPrefix("-1 3 4\n-2 1 \(UsageHistory.fnv1a("glm_moe_dsa"))\n"))
    #expect(try UsageHistory.parse(text, layers: 3, experts: 4) == c)
    #expect(UsageHistory.ranking(c).map { [$0.layer, $0.expert] } == [[1, 2], [2, 0]])
    #expect(throws: ColibriError.self) { try UsageHistory.parse(text, layers: 3, experts: 5) }
    #expect(UsageHistory.render(Array(repeating: [0, 0], count: 2)) == "")
    #expect(UsageHistory.fnv1a("") == 2_166_136_261)
}

// MARK: sampling

@Test func samplingIsGreedyAtZeroAndSeeded() {
    let logits: [Float] = [0.1, 3, -1, 2.9]
    var p = SamplingParameters()
    p.temperature = 0
    var s = Sampler(p)
    #expect(s.sample(logits) == 1)
    p.temperature = 1
    p.seed = 9
    var a = Sampler(p), b = Sampler(p)
    let ra = (0..<20).map { _ in a.sample(logits) }
    let rb = (0..<20).map { _ in b.sample(logits) }
    #expect(ra == rb)
    p.topP = 0.01   // nucleus of one token = greedy
    var c = Sampler(p)
    #expect((0..<10).allSatisfy { _ in c.sample(logits) == 1 })
}

// MARK: arguments

@Test func parsesCommandLine() throws {
    let a = try CLIArguments.parse(["serve", "--model", "/m", "--ram", "24", "--port", "9000",
                                    "--topk", "6", "--cors-origin", "*", "--no-think"])
    #expect(a.command == .serve)
    #expect(a.model == "/m")
    #expect(a.ramGB == 24)
    #expect(a.port == 9000)
    #expect(a.expertTopK == 6)
    #expect(a.corsOrigins == ["*"])
    #expect(a.thinking == false)
    let r = try CLIArguments.parse(["--model", "/m", "run", "hello", "world", "--temp", "0"])
    #expect(r.command == .run && r.prompt == ["hello", "world"] && r.temperature == 0)
    #expect(throws: CLIArguments.ParseError.unknownFlag("--nope")) { try CLIArguments.parse(["--nope"]) }
    #expect(throws: CLIArguments.ParseError.badValue("--port", "0")) { try CLIArguments.parse(["--port", "0"]) }
    #expect(throws: CLIArguments.ParseError.unknownCommand("fly")) { try CLIArguments.parse(["fly"]) }
}

// MARK: the model, end to end

/// Feeding tokens one at a time through the KV cache must give the same logits as
/// one prefill of the whole sequence: this checks rotary positions, the causal mask,
/// the compressed cache, and expert routing for batched rows.
@Test func incrementalDecodeMatchesPrefill() throws {
    let dir = try Tiny.write()
    defer { try? FileManager.default.removeItem(at: dir) }
    let model = try GLMModel(directory: dir, expertBudget: { _ in 1 << 30 })
    let tokens = [3, 17, 42, 8, 29]

    let full = try model.forward(tokens, cache: KVCache(config: model.config))
    let cache = KVCache(config: model.config)
    var step: [Float] = []
    for t in tokens { step = try model.forward([t], cache: cache) }
    #expect(cache.count == tokens.count)
    #expect(full.count == Tiny.vocab)
    let maxDiff = zip(full, step).map { abs($0 - $1) }.max() ?? 1
    #expect(maxDiff < 1e-3)
    #expect(full.allSatisfy { $0.isFinite })
}

/// Streaming every expert from disk (no cache) must compute exactly what a warm
/// cache computes, and the cache statistics must reflect hits and misses.
@Test func expertCacheDoesNotChangeResults() throws {
    let dir = try Tiny.write()
    defer { try? FileManager.default.removeItem(at: dir) }
    let cold = try GLMModel(directory: dir, expertBudget: { _ in 0 })
    let warm = try GLMModel(directory: dir, expertBudget: { _ in 1 << 30 })
    let tokens = [1, 2, 3, 4]
    let a = try cold.forward(tokens, cache: KVCache(config: cold.config))
    _ = try warm.forward(tokens, cache: KVCache(config: warm.config))
    let b = try warm.forward(tokens, cache: KVCache(config: warm.config))  // all hits now
    #expect(a == b)
    let s = warm.experts.snapshot()
    #expect(s.misses > 0 && s.hits >= s.misses)
    #expect(cold.experts.snapshot().cachedExperts == 0)
    #expect(s.cachedExperts > 0 && s.cachedBytes > 0)
    // Usage was recorded for sparse layers only.
    let u = warm.experts.usageSnapshot()
    #expect(u[0].allSatisfy { $0 == 0 })
    #expect(u[1].reduce(0, +) > 0)
}

/// Fewer experts per token still runs and changes the output.
@Test func expertTopKOverride() throws {
    let dir = try Tiny.write()
    defer { try? FileManager.default.removeItem(at: dir) }
    let m = try GLMModel(directory: dir, expertBudget: { _ in 1 << 30 })
    let a = try m.forward([5, 6], cache: KVCache(config: m.config))
    m.expertTopK = 1
    let b = try m.forward([5, 6], cache: KVCache(config: m.config))
    #expect(a != b)
}

/// A minimal tokenizer: one token per character code mod vocab.
final class ByteTokenizer: TextTokenizer {
    func encode(_ text: String) -> [Int] { text.unicodeScalars.map { 1 + Int($0.value) % (Tiny.vocab - 1) } }
    func decode(_ tokens: [Int]) -> String { String(tokens.map { Character(UnicodeScalar(UInt8(65 + $0 % 26))) }) }
    func chatPrompt(_ messages: [ChatMessage], thinking: Bool?) throws -> [Int] {
        encode(messages.map { "\($0.role):\($0.content)" }.joined(separator: "|"))
    }
    var eosTokens: [Int] { [] }
}

/// The engine reuses the KV cache across turns and honours max tokens and stops.
@Test func engineReusesPrefixAndStops() throws {
    let dir = try Tiny.write()
    defer { try? FileManager.default.removeItem(at: dir) }
    let model = try GLMModel(directory: dir, expertBudget: { _ in 1 << 30 })
    let engine = Engine(model: model, tokenizer: ByteTokenizer(), maxContext: 256)
    engine.saveUsage = false
    var o = GenerationOptions()
    o.sampling.temperature = 0
    o.sampling.maxTokens = 6
    var streamed = ""
    let r1 = try engine.generate(prompt: [5, 6, 7, 8], options: o, onText: { streamed += $0 })
    #expect(r1.completionTokens <= 6)
    #expect(r1.finishReason == .length || r1.finishReason == .stop)
    #expect(streamed == r1.text)
    let r2 = try engine.generate(prompt: [5, 6, 7, 8, 9], options: o)
    #expect(r2.reusedTokens == 4)
    // Greedy decoding is deterministic.
    let r3 = try engine.generate(prompt: [5, 6, 7, 8], options: o)
    #expect(r3.text == r1.text)
    // A stop string cuts the text before it.
    if r1.text.count > 1 {
        o.stop = [String(r1.text.dropFirst().prefix(1))]
        let r4 = try engine.generate(prompt: [5, 6, 7, 8], options: o)
        #expect(!r4.text.contains(o.stop[0]))
        #expect(r4.finishReason == .stop)
    }
}
