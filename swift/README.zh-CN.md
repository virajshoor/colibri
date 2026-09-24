# colibrì for Swift + MLX

[English](README.md) · 简体中文 · [繁體中文](README.zh-TW.md) · [Italiano](README.it.md) · [日本語](README.ja.md)

`swift/` 是 colibrì 面向 Apple Silicon 的原生前端：一个基于
[MLX Swift](https://github.com/ml-explore/mlx-swift) 和
[MLX Swift LM](https://github.com/ml-explore/mlx-swift-lm) 的小型 Swift
命令行工具。MLX 通过 Metal 在 GPU 上运行模型，并使用统一内存，因此 CPU 与
GPU 之间无需复制数据。

仓库包含两个引擎目录：

| 目录 | 语言 | 运行平台 | 适用场景 |
|---|---|---|---|
| [`c/`](../c) | C（+ CUDA、Metal、Vulkan 后端） | Linux、Windows、macOS | 使用 colibrì 自有格式、专家缓存和分层内存，从磁盘流式加载的前沿 MoE 模型（744B–2.8T） |
| `swift/` | Swift + MLX | Apple Silicon 上的 macOS 14+ | 能放进统一内存的 MLX 格式模型（例如 `mlx-community/*`） |

Swift 工具不读取 colibrì 的量化 safetensors 格式，也不从 SSD 流式加载专家。
放不进内存的模型请使用 C 引擎（`make` 和 `coli`，在 Mac 上加 `METAL=1`）。

## 环境要求

- Apple Silicon Mac，macOS 14 或更高版本
- Xcode 16 或更高版本（Swift 6.2 工具链）。需要完整的 Xcode，而不只是
  Command Line Tools，因为 MLX 自带的 Metal 着色器只能由 `xcodebuild` 编译。

## 构建

```bash
cd swift
xcodebuild -scheme colibri-swift -destination 'platform=macOS' -derivedDataPath .build/xcode build
```

可执行文件位于 `.build/xcode/Build/Products/Debug/colibri-swift`。
加上 `-configuration Release` 可得到优化构建（路径末尾变为 `Release/`）。

`swift build` 也能编译代码，但生成的二进制在运行时无法加载 MLX 的 Metal
库，只适合用来检查代码能否编译。

## 运行

```bash
# 单条提示，流式输出到 stdout
.build/xcode/Build/Products/Debug/colibri-swift "用两句话解释 mixture-of-experts"

# 交互式聊天（空行或 Ctrl-D 退出）
.build/xcode/Build/Products/Debug/colibri-swift --model mlx-community/Qwen3-4B-4bit
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `-m`, `--model` | `mlx-community/Qwen3-4B-4bit` | Hugging Face 模型 id，或包含 MLX 权重的本地目录 |
| `-n`, `--max-tokens` | `512` | 每次回复的最大 token 数 |
| `-t`, `--temp` | `0.6` | 采样温度（`0` = greedy） |
| `-s`, `--system` | 无 | 系统提示词 |
| `-h`, `--help` | | 显示用法 |

按 id 指定的模型会在首次使用时从 Hugging Face 下载并缓存；本地目录直接从
磁盘加载，无需联网。MLX Swift LM 支持的任何架构都可使用，包括 Qwen、Llama、
Gemma、Mistral、Phi 和 DeepSeek 系列。要把 Hugging Face checkpoint 转成 MLX
格式，请使用 Python `mlx-lm` 包中的 `mlx_lm.convert`。

## 测试

```bash
cd swift
swift test
```

测试覆盖参数解析，位于不依赖 MLX 的 `ColibriOptions` target 中。

## 目录结构

```
swift/
├── Package.swift                     SwiftPM 清单（mlx-swift-lm、swift-huggingface、swift-transformers）
├── Sources/ColibriOptions/           命令行解析（不依赖 MLX）
├── Sources/ColibriSwift/main.swift   模型加载与流式聊天
└── Tests/ColibriOptionsTests/        swift-testing 测试
```
