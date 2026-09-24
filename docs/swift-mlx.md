# Swift + MLX front end (Apple Silicon)

The repository has two engine trees. `c/` is the C engine that streams frontier
MoE models from disk on Linux, Windows and macOS (with the [Metal backend](metal.md)
on a Mac). `swift/` is a small Swift command-line tool built on
[MLX Swift](https://github.com/ml-explore/mlx-swift) and
[MLX Swift LM](https://github.com/ml-explore/mlx-swift-lm) that runs
MLX-format models (for example `mlx-community/*`) that fit in unified memory.

The Swift tool does not read colibrì's quantized safetensors formats and does not stream experts
from SSD. Use the C engine for models larger than RAM.

## Build and run

Requires an Apple Silicon Mac on macOS 14+ and Xcode 16+. MLX's Metal shaders
are only compiled by `xcodebuild`, so build with it rather than `swift build`:

```bash
cd swift
xcodebuild -scheme colibri-swift -destination 'platform=macOS' -derivedDataPath .build/xcode build
.build/xcode/Build/Products/Debug/colibri-swift --model mlx-community/Qwen3-4B-4bit "Hello"
```

| Flag | Default | Meaning |
|---|---|---|
| `-m`, `--model` | `mlx-community/Qwen3-4B-4bit` | Hugging Face id, or a local directory with MLX weights |
| `-n`, `--max-tokens` | `512` | Maximum tokens per reply |
| `-t`, `--temp` | `0.6` | Sampling temperature (`0` = greedy) |
| `-s`, `--system` | none | System prompt |

With no prompt the tool starts an interactive chat. Run `swift test` in
`swift/` for the argument-parsing tests. Full details are in
[`swift/README.md`](https://github.com/JustVugg/colibri/blob/main/swift/README.md),
also available in 简体中文, 繁體中文, Italiano and 日本語.
