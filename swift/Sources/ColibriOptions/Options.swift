/// Command-line options for `colibri-swift`.
///
/// This target has no MLX dependency so `swift test` can check the parser
/// without compiling MLX's Metal shaders.
public struct Options: Equatable, Sendable {
    /// Model loaded when `--model` is not given: a small 4-bit MLX checkpoint
    /// that fits on any Apple Silicon Mac.
    public static let defaultModel = "mlx-community/Qwen3-4B-4bit"

    /// Text printed for `--help` and after a parse error.
    public static let usage = """
        usage: colibri-swift [--model <hf-id|dir>] [--max-tokens N] [--temp T] [--system TEXT] [prompt...]
          With no prompt, starts an interactive chat (empty line or EOF quits).
        """

    /// Hugging Face model id, or a path to a local directory of MLX weights.
    public var model = Options.defaultModel
    /// Upper bound on generated tokens per reply.
    public var maxTokens = 512
    /// Sampling temperature; 0 always picks the most likely token (greedy).
    public var temperature: Float = 0.6
    /// Optional system prompt for the chat session.
    public var system: String? = nil
    /// One-shot prompt. `nil` means interactive chat.
    public var prompt: String? = nil

    public init() {}

    /// Why parsing stopped. `.help` is not an error for the user; the caller
    /// prints usage and exits 0.
    public enum ParseError: Error, Equatable {
        case help
        case missingValue(String)       // flag given as the last argument
        case badValue(String, String)   // flag, rejected value
        case unknownFlag(String)
    }

    /// Parses arguments (without the program name). Flags may appear anywhere;
    /// every non-flag word is joined with spaces into the prompt. `--` ends
    /// flag parsing so a prompt can start with `-`.
    public static func parse(_ args: [String]) throws(ParseError) -> Options {
        var o = Options()
        var words: [String] = []  // prompt words, in order
        var i = 0

        // Advances past the flag and returns its value, or fails if the flag
        // was the last argument.
        func value(_ flag: String) throws(ParseError) -> String {
            i += 1
            guard i < args.count else { throw .missingValue(flag) }
            return args[i]
        }

        while i < args.count {
            let a = args[i]
            switch a {
            case "-h", "--help": throw .help
            case "-m", "--model": o.model = try value(a)
            case "-n", "--max-tokens":
                let v = try value(a)
                guard let n = Int(v), n > 0 else { throw .badValue(a, v) }
                o.maxTokens = n
            case "-t", "--temp":
                let v = try value(a)
                guard let t = Float(v), t >= 0 else { throw .badValue(a, v) }
                o.temperature = t
            case "-s", "--system": o.system = try value(a)
            case "--":
                // Everything after `--` is prompt text, even if it looks like a flag.
                words += args[(i + 1)...]
                i = args.count
            default:
                // A lone "-" is treated as a word; anything else starting with "-" is a typo.
                if a.hasPrefix("-") && a.count > 1 { throw .unknownFlag(a) }
                words.append(a)
            }
            i += 1
        }
        if !words.isEmpty { o.prompt = words.joined(separator: " ") }
        return o
    }
}
