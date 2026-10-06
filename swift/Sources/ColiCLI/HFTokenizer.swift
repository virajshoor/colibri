// The model's tokenizer and chat template, read from tokenizer.json and
// tokenizer_config.json in the model directory by swift-transformers. Nothing is
// downloaded: the folder is loaded locally.

import ColibriCore
import Foundation
import Tokenizers

final class HFTokenizer: TextTokenizer {
    let tokenizer: any Tokenizer

    init(folder: URL) async throws {
        tokenizer = try await AutoTokenizer.from(modelFolder: folder)
    }

    func encode(_ text: String) -> [Int] { tokenizer.encode(text: text) }

    /// Special tokens (role markers, end-of-turn) are not shown to the user.
    func decode(_ tokens: [Int]) -> String { tokenizer.decode(tokens: tokens, skipSpecialTokens: true) }

    var eosTokens: [Int] { tokenizer.eosTokenId.map { [$0] } ?? [] }

    func chatPrompt(_ messages: [ChatMessage], thinking: Bool?) throws -> [Int] {
        let msgs: [Message] = messages.map { ["role": $0.role, "content": $0.content] }
        var context: [String: any Sendable]? = nil
        if let thinking { context = ["enable_thinking": thinking] }
        return try tokenizer.applyChatTemplate(
            messages: msgs, chatTemplate: nil, addGenerationPrompt: true, truncation: false,
            maxLength: nil, tools: nil, additionalContext: context)
    }
}
