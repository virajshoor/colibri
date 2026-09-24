// colibri-swift: load an MLX-format language model and chat with it.
//
// MLX runs the model on the Apple GPU through Metal, in unified memory.
// Model download, tokenizer and chat template come from mlx-swift-lm and its
// Hugging Face integration; this file only wires them to the command line.

import ColibriOptions
import Foundation
import HuggingFace      // Hugging Face Hub client used by the load macro
import MLXHuggingFace   // #huggingFaceLoadModelContainer
import MLXLLM           // registers the LLM architectures (Qwen, Llama, Gemma, ...)
import MLXLMCommon      // ModelConfiguration, ChatSession, GenerateParameters
import Tokenizers       // tokenizer implementation used by the load macro

// Parse arguments. `--help` exits 0; any other parse error exits 2 (usage error).
let opts: Options
do {
    opts = try Options.parse(Array(CommandLine.arguments.dropFirst()))
} catch .help {
    print(Options.usage)
    exit(0)
} catch {
    FileHandle.standardError.write(Data("colibri-swift: \(error)\n\(Options.usage)\n".utf8))
    exit(2)
}

// A local directory loads MLX weights from disk; anything else is a Hugging Face id.
var isDir: ObjCBool = false
let configuration =
    FileManager.default.fileExists(atPath: opts.model, isDirectory: &isDir) && isDir.boolValue
    ? ModelConfiguration(directory: URL(fileURLWithPath: opts.model))
    : ModelConfiguration(id: opts.model)

// Downloads (or reuses the cached copy of) the weights and tokenizer, then
// builds the model. The container serializes access to the model across tasks.
let container = try await #huggingFaceLoadModelContainer(configuration: configuration)

// A ChatSession keeps the conversation and its KV cache, so each turn only
// processes the new message instead of the whole history.
let session = ChatSession(
    container,
    instructions: opts.system,
    generateParameters: GenerateParameters(maxTokens: opts.maxTokens, temperature: opts.temperature))

/// Sends one user message and prints the reply as it is generated.
func reply(_ prompt: String) async throws {
    for try await chunk in session.streamResponse(to: prompt) {
        print(chunk, terminator: "")
        fflush(stdout)  // show text immediately instead of when the line buffer fills
    }
    print()
}

if let prompt = opts.prompt {
    // One-shot mode: answer the prompt from the command line and exit.
    try await reply(prompt)
} else {
    // Interactive mode: one turn per line until an empty line or EOF (Ctrl-D).
    while true {
        print("> ", terminator: "")
        fflush(stdout)
        guard let line = readLine(), !line.isEmpty else { break }
        try await reply(line)
    }
}
