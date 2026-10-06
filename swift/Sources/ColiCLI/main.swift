// `coli`: colibrì's command line, entirely in Swift.
//
//   coli run "prompt"    one-shot generation
//   coli chat            interactive chat
//   coli serve           OpenAI-compatible HTTP API
//   coli info / plan     model and memory summary
//
// The model directory is a colibrì-converted GLM / DeepSeek-V3-family checkpoint
// (safetensors with `.qs` scale sidecars, plus config.json and tokenizer files).
// Dense weights load into RAM; routed experts stream from SSD on demand.

import ColibriCore
import ColibriServer
import Foundation

/// Prints to standard error (status lines, statistics), keeping stdout for the reply.
func note(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

func fail(_ s: String, code: Int32 = 1) -> Never {
    note("coli: \(s)")
    exit(code)
}

let args: CLIArguments
do {
    args = try CLIArguments.parse(Array(CommandLine.arguments.dropFirst()))
} catch {
    note("coli: \(error)\n\n\(CLIArguments.usage)")
    exit(2)
}
if args.command == .help {
    print(CLIArguments.usage)
    exit(0)
}
guard let modelPath = args.model else { fail("no model directory: pass --model DIR or set COLI_MODEL", code: 2) }
let modelDir = URL(fileURLWithPath: (modelPath as NSString).expandingTildeInPath)

// MARK: info / plan (no weights loaded)

/// Sums tensor bytes by role, reading only safetensors headers.
func diskSummary(_ shards: ShardSet) -> (dense: Int64, experts: Int64, formats: [String: Int]) {
    var dense: Int64 = 0, experts: Int64 = 0
    var formats: [String: Int] = [:]
    for (name, t) in shards.tensors {
        let isExpert = name.contains(".mlp.experts.")
        if isExpert { experts += Int64(t.byteCount) } else { dense += Int64(t.byteCount) }
        if name.hasSuffix(".qs") || isExpert { continue }
        let key = shards.has(name + ".qs") ? "quantized" : t.dtype.rawValue
        formats[key, default: 0] += 1
    }
    return (dense, experts, formats)
}

if args.command == .info || args.command == .plan {
    do {
        let c = try ModelConfig.load(directory: modelDir)
        let shards = try ShardSet(directory: modelDir)
        let d = diskSummary(shards)
        let store = ExpertStore(shards: shards, config: c, budget: 0)
        let plan = MemoryPlan(config: c, ramGB: args.ramGB, denseBytes: d.dense,
                              expertBytes: store.bytesPerExpert(), context: args.context)
        if args.json {
            var o = plan.json
            if args.command == .info {
                o["model_type"] = c.modelType
                o["layers"] = c.layers
                o["experts_per_layer"] = c.experts
                o["experts_per_token"] = c.topK
                o["hidden"] = c.hidden
                o["vocab"] = c.vocab
                o["shards"] = shards.files.count
                o["tensors"] = shards.tensors.count
                o["disk_dense_bytes"] = d.dense
                o["disk_expert_bytes"] = d.experts
                o["ram_total"] = HostMemory.total
                o["ram_available"] = HostMemory.available
            }
            print(JSON.string(o))
        } else {
            if args.command == .info {
                print("""
                    model        \(modelDir.path)
                    type         \(c.modelType.isEmpty ? "?" : c.modelType)  \(c.layers) layers, hidden \(c.hidden), vocab \(c.vocab)
                    attention    MLA: \(c.heads) heads, q_lora \(c.qLora), kv_lora \(c.kvLora), rope \(c.qkRope)
                    experts      \(c.experts) per layer, \(c.topK) per token, \(c.sharedExperts) shared, dense layers \(c.firstDense)
                    files        \(shards.files.count) shards, \(shards.tensors.count) tensors
                    on disk      dense \(formatBytes(d.dense)), experts \(formatBytes(d.experts))
                    formats      \(d.formats.sorted { $0.key < $1.key }.map { "\($0.key)×\($0.value)" }.joined(separator: ", "))
                    host RAM     \(formatBytes(HostMemory.total)) total, \(formatBytes(HostMemory.available)) available
                    """)
            }
            print(plan.describe())
            let perToken = Int64(c.topK * (c.layers - c.firstDense)) * plan.expertBytes
            print("expert reads   up to \(formatBytes(perToken)) per token on a cold cache")
        }
    } catch {
        fail("\(error)")
    }
    exit(0)
}

// MARK: load the model

let t0 = monotonicSeconds()
let model: GLMModel
let tokenizer: HFTokenizer
do {
    note("[colibri] loading \(modelDir.path)")
    model = try GLMModel(directory: modelDir, expertBudget: { dense in
        let c = try? ModelConfig.load(directory: modelDir)
        let budget = args.ramGB > 0 ? Int64(args.ramGB * 1e9) : MemoryPlan.defaultBudget()
        let kv = c.map { MemoryPlan.kvBytes($0, context: args.context) } ?? 0
        return max(0, budget - dense - kv - 1_000_000_000)
    }, progress: { note("[colibri] \($0)") })
    tokenizer = try await HFTokenizer(folder: modelDir)
} catch {
    fail("\(error)")
}
model.expertTopK = args.expertTopK
model.expertTopP = args.expertTopP
note("[colibri] expert cache \(formatBytes(model.experts.budget)), loaded in \(String(format: "%.1f", monotonicSeconds() - t0)) s")

// Pin the experts this model used most in earlier sessions.
do {
    let c = model.config
    var history: [[UInt32]]? = nil
    switch args.pin {
    case "none": break
    case "auto": history = try UsageHistory.load(directory: modelDir, layers: c.layers, experts: c.experts)
    default:
        let text = try String(contentsOfFile: args.pin, encoding: .utf8)
        history = try UsageHistory.parse(text, layers: c.layers, experts: c.experts)
    }
    if let history {
        model.experts.setUsage(history)
        let pinBytes = args.pinGB.map { Int64($0 * 1e9) } ?? model.experts.budget / 2
        if pinBytes > 0 {
            let tp = monotonicSeconds()
            let n = try model.experts.pin(UsageHistory.ranking(history), maxBytes: min(pinBytes, model.experts.budget))
            note("[colibri] pinned \(n) experts from usage history in \(String(format: "%.1f", monotonicSeconds() - tp)) s")
        }
    }
} catch {
    note("[colibri] usage history not used: \(error)")
}

let engine = Engine(model: model, tokenizer: tokenizer, maxContext: args.context)

var options = GenerationOptions()
options.sampling.temperature = args.temperature ?? 1.0
options.sampling.topP = args.topP ?? 0.95
options.sampling.maxTokens = args.maxTokens ?? 2048
options.sampling.seed = args.seed

/// One-line statistics after a reply.
func statsLine(_ r: GenerationResult) -> String {
    let e = r.experts
    let rate = String(format: "%.2f", r.tokensPerSecond)
    let prefill = String(format: "%.1f", r.prefillSeconds)
    let hit = String(format: "%.0f", e.hitRate * 100)
    let read = String(format: "%.1f", e.readSeconds)
    return "[\(r.completionTokens) tok, \(rate) tok/s, prefill \(prefill) s (\(r.reusedTokens) cached), "
        + "expert hit \(hit)%, read \(formatBytes(e.bytesRead)) in \(read) s]"
}

func streamOut(_ s: String) {
    FileHandle.standardOutput.write(Data(s.utf8))
}

// MARK: commands

switch args.command {
case .run:
    let prompt = args.prompt.joined(separator: " ")
    guard !prompt.isEmpty else { fail("run needs a prompt", code: 2) }
    var messages: [ChatMessage] = []
    if let s = args.system { messages.append(ChatMessage(role: "system", content: s)) }
    messages.append(ChatMessage(role: "user", content: prompt))
    do {
        let r = try engine.chat(messages, thinking: args.thinking, options: options, onText: streamOut)
        print()
        note(statsLine(r))
    } catch {
        fail("\(error)")
    }

case .chat:
    var history: [ChatMessage] = []
    if let s = args.system { history.append(ChatMessage(role: "system", content: s)) }
    note("[colibri] chat ready — /reset clears the conversation, /exit or Ctrl-D quits")
    while true {
        streamOut("\n> ")
        guard let line = readLine() else { break }
        let text = line.trimmingCharacters(in: .whitespaces)
        if text.isEmpty { continue }
        if text == "/exit" || text == "/quit" { break }
        if text == "/reset" {
            history = history.filter { $0.role == "system" }
            engine.reset()
            note("[colibri] conversation cleared")
            continue
        }
        history.append(ChatMessage(role: "user", content: text))
        do {
            let r = try engine.chat(history, thinking: args.thinking, options: options, onText: streamOut)
            print()
            note(statsLine(r))
            history.append(ChatMessage(role: "assistant", content: r.text))
        } catch {
            note("coli: \(error)")
            history.removeLast()
        }
    }

case .serve:
    var sc = ServerConfig()
    sc.host = args.host
    sc.port = args.port
    sc.apiKey = args.apiKey
    sc.corsOrigins = args.corsOrigins
    sc.allowedHosts = args.allowedHosts
    sc.maxQueue = args.maxQueue
    sc.webRoot = args.webRoot
    sc.defaults.temperature = options.sampling.temperature
    sc.defaults.topP = options.sampling.topP
    sc.defaults.maxTokens = options.sampling.maxTokens
    let id = args.modelID ?? modelDir.lastPathComponent
    let server = OpenAIServer(config: sc, backend: EngineBackend(engine: engine, modelID: id))
    do {
        try server.run { url in note("[colibri] serving \(id) at \(url)/v1 (OpenAI-compatible)") }
    } catch {
        fail("\(error)")
    }

case .info, .plan, .help:
    break
}
