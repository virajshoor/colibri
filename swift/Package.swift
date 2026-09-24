// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "colibri-swift",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "colibri-swift", targets: ["ColibriSwift"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.32.3")),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
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
        .testTarget(name: "ColibriOptionsTests", dependencies: ["ColibriOptions"]),
    ]
)
