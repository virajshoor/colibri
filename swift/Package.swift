// swift-tools-version: 6.2
// SwiftPM manifest for colibrì's Swift tree.
//
// Two programs live here and share no code with ../c:
//
//   coli            the streaming MoE engine: reads colibrì's quantized safetensors
//                   containers, streams routed experts from SSD into a RAM cache, and
//                   serves an OpenAI-compatible HTTP API. Pure Swift + Accelerate.
//   colibri-swift   a small chat tool for MLX-format models that fit in unified memory,
//                   built on MLX Swift LM.
//
// The engine targets compile in the Swift 5 language mode: they share raw buffers
// between worker threads on purpose (each thread writes a disjoint slice), which the
// Swift 6 data-race checker cannot prove safe and would reject.
import PackageDescription

let swift5: [SwiftSetting] = [.swiftLanguageMode(.v5)]

let package = Package(
    name: "colibri-swift",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "coli", targets: ["ColiCLI"]),
        .executable(name: "colibri-swift", targets: ["ColibriSwift"]),
        .library(name: "ColibriCore", targets: ["ColibriCore"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.32.3")),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        // ---- streaming engine (no MLX, no C) ----
        // Tensor storage, quantized formats, expert cache, model forward pass, sampling.
        .target(name: "ColibriCore", swiftSettings: swift5),
        // HTTP/1.1 server and the OpenAI-compatible routes, written against a protocol so
        // it tests without a model.
        .target(name: "ColibriServer", dependencies: ["ColibriCore"], swiftSettings: swift5),
        // The `coli` command: run, chat, serve, info, plan.
        .executableTarget(
            name: "ColiCLI",
            dependencies: [
                "ColibriCore",
                "ColibriServer",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: swift5),

        // ---- MLX chat tool ----
        // Argument parsing kept free of MLX so it tests without building Metal.
        .target(name: "ColibriOptions"),
        .executableTarget(
            name: "ColibriSwift",
            dependencies: [
                "ColibriOptions",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]),

        // ---- tests ----
        .testTarget(name: "ColibriOptionsTests", dependencies: ["ColibriOptions"]),
        .testTarget(name: "ColibriCoreTests", dependencies: ["ColibriCore"], swiftSettings: swift5),
        .testTarget(
            name: "ColibriServerTests", dependencies: ["ColibriServer", "ColibriCore"],
            swiftSettings: swift5),
    ]
)
