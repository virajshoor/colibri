// Tests for the command-line parser. They run with `swift test` and need no
// GPU or model download because ColibriOptions does not depend on MLX.

import Testing

@testable import ColibriOptions

/// No arguments gives the documented defaults and interactive mode.
@Test func defaults() throws {
    let o = try Options.parse([])
    #expect(o == Options())
    #expect(o.prompt == nil)
}

/// Short and long flags set their fields; remaining words form the prompt.
@Test func flagsAndPrompt() throws {
    let o = try Options.parse(["-m", "/models/q", "--max-tokens", "64", "-t", "0", "hello", "world"])
    #expect(o.model == "/models/q")
    #expect(o.maxTokens == 64)
    #expect(o.temperature == 0)
    #expect(o.prompt == "hello world")
}

/// After `--`, words that look like flags are kept as prompt text.
@Test func doubleDashKeepsDashes() throws {
    #expect(try Options.parse(["--", "-n", "x"]).prompt == "-n x")
}

/// Each kind of bad input maps to its own error.
@Test func errors() {
    #expect(throws: Options.ParseError.help) { try Options.parse(["--help"]) }
    #expect(throws: Options.ParseError.missingValue("--model")) { try Options.parse(["--model"]) }
    #expect(throws: Options.ParseError.badValue("-n", "0")) { try Options.parse(["-n", "0"]) }
    #expect(throws: Options.ParseError.unknownFlag("--nope")) { try Options.parse(["--nope"]) }
}
