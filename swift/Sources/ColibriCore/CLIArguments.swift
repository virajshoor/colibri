// Command-line arguments of `coli`, parsed without any dependency so they are tested
// directly. Flags may come before or after the subcommand; words that are not flags
// form the prompt of `coli run`.

import Foundation

public struct CLIArguments: Equatable, Sendable {
    public enum Command: String, Sendable, CaseIterable {
        case run, chat, serve, info, plan, help
    }

    public var command = Command.help
    public var model: String? = ProcessInfo.processInfo.environment["COLI_MODEL"]
    public var ramGB: Double = 0           // 0 = 85% of available memory
    public var context = 8192
    public var maxTokens: Int? = nil
    public var temperature: Float? = nil
    public var topP: Float? = nil           // sampling nucleus
    public var seed: UInt64? = nil
    public var expertTopK: Int? = nil       // experts per token (fewer = less I/O)
    public var expertTopP: Float = 0        // adaptive expert selection
    public var pin = "auto"                 // auto | none | path to a usage history
    public var pinGB: Double? = nil         // RAM for pinned experts (default: half the cache)
    public var system: String? = nil
    public var thinking: Bool? = nil
    public var json = false
    public var prompt: [String] = []
    // serve
    public var host = "127.0.0.1"
    public var port = 8000
    public var modelID: String? = ProcessInfo.processInfo.environment["COLI_MODEL_ID"]
    public var apiKey: String? = ProcessInfo.processInfo.environment["COLI_API_KEY"]
    public var corsOrigins: [String] = []
    public var allowedHosts: [String] = []
    public var maxQueue = 8
    public var webRoot: String? = nil

    public init() {}

    public static let usage = """
        colibrì — tiny engine, immense model. Native Swift engine for Apple Silicon.

        usage: coli <command> [options]

        commands:
          run "prompt"      one-shot generation, streamed to stdout
          chat              interactive chat (/reset clears, /exit or Ctrl-D quits)
          serve             OpenAI-compatible HTTP API
          info              model, formats, memory and disk summary
          plan              RAM plan: resident weights, KV cache, expert cache

        model and memory:
          --model DIR       converted model directory (or COLI_MODEL)
          --ram GB          RAM budget; the expert cache gets what is left (default: 85% of available)
          --ctx N           maximum context in tokens (default 8192)
          --pin MODE        auto (pin experts from .coli_usage), none, or a history file path
          --pin-gb GB       RAM for pinned experts (default: half the expert cache)
          --topk N          experts per token (default: the model's)
          --topp P          adaptive experts: smallest set reaching this routing mass

        generation:
          --ngen N          maximum reply tokens (default 2048)
          --temp T          temperature (default 1.0; 0 = greedy)
          --top-p P         nucleus sampling (default 0.95)
          --seed N          reproducible sampling
          --system TEXT     system prompt
          --think, --no-think   toggle reasoning in templates that support it

        serve:
          --host H --port P (default 127.0.0.1:8000)   --model-id ID   --api-key KEY
          --cors-origin O   --allowed-host H   --max-queue N   --web-root DIR
          --json            machine-readable output for info and plan
        """

    public enum ParseError: Error, Equatable, CustomStringConvertible {
        case missingValue(String)
        case badValue(String, String)
        case unknownFlag(String)
        case unknownCommand(String)

        public var description: String {
            switch self {
            case .missingValue(let f): return "\(f) needs a value"
            case .badValue(let f, let v): return "bad value for \(f): \(v)"
            case .unknownFlag(let f): return "unknown option \(f)"
            case .unknownCommand(let c): return "unknown command \(c)"
            }
        }
    }

    public static func parse(_ args: [String]) throws -> CLIArguments {
        var a = CLIArguments()
        var sawCommand = false
        var i = 0
        func value(_ flag: String) throws -> String {
            i += 1
            guard i < args.count else { throw ParseError.missingValue(flag) }
            return args[i]
        }
        func number<T: LosslessStringConvertible>(_ flag: String, _ ok: (T) -> Bool = { _ in true }) throws -> T {
            let v = try value(flag)
            guard let n = T(v), ok(n) else { throw ParseError.badValue(flag, v) }
            return n
        }
        while i < args.count {
            let s = args[i]
            switch s {
            case "-h", "--help": a.command = .help; return a
            case "--model", "-m": a.model = try value(s)
            case "--ram": a.ramGB = try number(s) { (x: Double) in x >= 0 }
            case "--ctx": a.context = try number(s) { (x: Int) in x >= 16 }
            case "--ngen", "--max-tokens", "-n": a.maxTokens = try number(s) { (x: Int) in x > 0 }
            case "--temp", "-t": a.temperature = try number(s) { (x: Float) in x >= 0 }
            case "--top-p": a.topP = try number(s) { (x: Float) in x > 0 && x <= 1 }
            case "--seed": a.seed = try number(s) as UInt64
            case "--topk": a.expertTopK = try number(s) { (x: Int) in x > 0 }
            case "--topp": a.expertTopP = try number(s) { (x: Float) in x >= 0 && x <= 1 }
            case "--pin": a.pin = try value(s)
            case "--pin-gb": a.pinGB = try number(s) { (x: Double) in x >= 0 }
            case "--system", "-s": a.system = try value(s)
            case "--think": a.thinking = true
            case "--no-think": a.thinking = false
            case "--json": a.json = true
            case "--host": a.host = try value(s)
            case "--port": a.port = try number(s) { (x: Int) in x > 0 && x < 65536 }
            case "--model-id": a.modelID = try value(s)
            case "--api-key": a.apiKey = try value(s)
            case "--cors-origin": a.corsOrigins.append(try value(s))
            case "--allowed-host": a.allowedHosts.append(try value(s))
            case "--max-queue": a.maxQueue = try number(s) { (x: Int) in x >= 0 }
            case "--web-root": a.webRoot = try value(s)
            case "--":
                a.prompt += args[(i + 1)...]
                i = args.count
            default:
                if s.hasPrefix("-") && s.count > 1 { throw ParseError.unknownFlag(s) }
                if !sawCommand {
                    guard let c = Command(rawValue: s) else { throw ParseError.unknownCommand(s) }
                    a.command = c
                    sawCommand = true
                } else {
                    a.prompt.append(s)
                }
            }
            i += 1
        }
        return a
    }
}
