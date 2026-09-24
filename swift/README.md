# colibrì for Swift + MLX

English · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [Italiano](README.it.md) · [日本語](README.ja.md)

`swift/` is colibrì's native Apple Silicon front end. It is a small Swift
command-line tool built on [MLX Swift](https://github.com/ml-explore/mlx-swift)
and [MLX Swift LM](https://github.com/ml-explore/mlx-swift-lm). MLX runs the
model on the GPU through Metal and uses unified memory, so nothing is copied
between CPU and GPU.

The repository has two engine trees:

| Folder | Language | Runs | Use it for |
|---|---|---|---|
| [`c/`](../c) | C (+ CUDA, Metal, Vulkan backends) | Linux, Windows, macOS | Frontier MoE models (744B–2.8T) streamed from disk with colibrì's own formats, expert cache and tiered memory |
| `swift/` | Swift + MLX | macOS 14+ on Apple Silicon | Models published in MLX format (for example `mlx-community/*`) that fit in unified memory |

The Swift tool does not read colibrì's quantized safetensors formats and does not stream
experts from SSD. For models that do not fit in RAM, use the C engine
(`make` and `coli`, with `METAL=1` on a Mac).

## Requirements

- Apple Silicon Mac, macOS 14 or newer
- Xcode 16 or newer (Swift 6.2 toolchain). The full Xcode is needed, not just
  the Command Line Tools, because MLX ships Metal shaders that only
  `xcodebuild` compiles.

## Build

```bash
cd swift
xcodebuild -scheme colibri-swift -destination 'platform=macOS' -derivedDataPath .build/xcode build
```

The binary is written to `.build/xcode/Build/Products/Debug/colibri-swift`.
Add `-configuration Release` for an optimized build (the path then ends in
`Release/`).

`swift build` also compiles the code, but the resulting binary cannot load
MLX's Metal library at run time. Use it only to check that the code compiles.

## Run

```bash
# One prompt, streamed to stdout
.build/xcode/Build/Products/Debug/colibri-swift "Explain mixture-of-experts in two sentences"

# Interactive chat (empty line or Ctrl-D quits)
.build/xcode/Build/Products/Debug/colibri-swift --model mlx-community/Qwen3-4B-4bit
```

| Flag | Default | Meaning |
|---|---|---|
| `-m`, `--model` | `mlx-community/Qwen3-4B-4bit` | Hugging Face model id, or a local directory with MLX weights |
| `-n`, `--max-tokens` | `512` | Maximum tokens per reply |
| `-t`, `--temp` | `0.6` | Sampling temperature (`0` = greedy) |
| `-s`, `--system` | none | System prompt |
| `-h`, `--help` | | Print usage |

Models given by id are downloaded from Hugging Face on first use and cached.
A local directory is loaded from disk without network access. Any architecture
supported by MLX Swift LM works, including Qwen, Llama, Gemma, Mistral, Phi and
DeepSeek families. To convert a Hugging Face checkpoint to MLX format, use
`mlx_lm.convert` from the Python `mlx-lm` package.

## Test

```bash
cd swift
swift test
```

The tests cover argument parsing. They live in the `ColibriOptions` target,
which does not depend on MLX.

## Layout

```
swift/
├── Package.swift                     SwiftPM manifest (mlx-swift-lm, swift-huggingface, swift-transformers)
├── Sources/ColibriOptions/           command-line parsing (no MLX dependency)
├── Sources/ColibriSwift/main.swift   model loading and streaming chat
└── Tests/ColibriOptionsTests/        swift-testing tests
```
