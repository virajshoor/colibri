# colibrì for Swift + MLX

[English](README.md) · [简体中文](README.zh-CN.md) · 繁體中文 · [Italiano](README.it.md) · [日本語](README.ja.md)

`swift/` 是 colibrì 針對 Apple Silicon 的原生前端：一個以
[MLX Swift](https://github.com/ml-explore/mlx-swift) 與
[MLX Swift LM](https://github.com/ml-explore/mlx-swift-lm) 為基礎的小型 Swift
命令列工具。MLX 透過 Metal 在 GPU 上執行模型，並使用統一記憶體，因此 CPU 與
GPU 之間不需要複製資料。

儲存庫包含兩個引擎目錄：

| 目錄 | 語言 | 執行平台 | 適用情境 |
|---|---|---|---|
| [`c/`](../c) | C（+ CUDA、Metal、Vulkan 後端） | Linux、Windows、macOS | 使用 colibrì 自有格式、專家快取與分層記憶體，從磁碟串流載入的前沿 MoE 模型（744B–2.8T） |
| `swift/` | Swift + MLX | Apple Silicon 上的 macOS 14+ | 可放進統一記憶體的 MLX 格式模型（例如 `mlx-community/*`） |

Swift 工具不讀取 colibrì 的量化 safetensors 格式，也不從 SSD 串流載入專家。
放不進記憶體的模型請使用 C 引擎（`make` 與 `coli`，在 Mac 上加 `METAL=1`）。

## 環境需求

- Apple Silicon Mac，macOS 14 或更新版本
- Xcode 16 或更新版本（Swift 6.2 工具鏈）。需要完整的 Xcode，而不只是
  Command Line Tools，因為 MLX 內附的 Metal 著色器只能由 `xcodebuild` 編譯。

## 建置

```bash
cd swift
xcodebuild -scheme colibri-swift -destination 'platform=macOS' -derivedDataPath .build/xcode build
```

執行檔位於 `.build/xcode/Build/Products/Debug/colibri-swift`。
加上 `-configuration Release` 可取得最佳化建置（路徑結尾變為 `Release/`）。

`swift build` 也能編譯程式碼，但產生的二進位檔在執行時無法載入 MLX 的 Metal
函式庫，只適合用來檢查程式碼能否編譯。

## 執行

```bash
# 單一提示，串流輸出到 stdout
.build/xcode/Build/Products/Debug/colibri-swift "用兩句話解釋 mixture-of-experts"

# 互動式聊天（空行或 Ctrl-D 結束）
.build/xcode/Build/Products/Debug/colibri-swift --model mlx-community/Qwen3-4B-4bit
```

| 參數 | 預設值 | 說明 |
|---|---|---|
| `-m`, `--model` | `mlx-community/Qwen3-4B-4bit` | Hugging Face 模型 id，或含有 MLX 權重的本機目錄 |
| `-n`, `--max-tokens` | `512` | 每次回覆的最大 token 數 |
| `-t`, `--temp` | `0.6` | 取樣溫度（`0` = greedy） |
| `-s`, `--system` | 無 | 系統提示詞 |
| `-h`, `--help` | | 顯示用法 |

以 id 指定的模型會在第一次使用時從 Hugging Face 下載並快取；本機目錄直接從
磁碟載入，不需連網。MLX Swift LM 支援的任何架構都能使用，包括 Qwen、Llama、
Gemma、Mistral、Phi 與 DeepSeek 系列。要把 Hugging Face checkpoint 轉成 MLX
格式，請使用 Python `mlx-lm` 套件中的 `mlx_lm.convert`。

## 測試

```bash
cd swift
swift test
```

測試涵蓋參數解析，位於不依賴 MLX 的 `ColibriOptions` target 中。

## 目錄結構

```
swift/
├── Package.swift                     SwiftPM 清單（mlx-swift-lm、swift-huggingface、swift-transformers）
├── Sources/ColibriOptions/           命令列解析（不依賴 MLX）
├── Sources/ColibriSwift/main.swift   模型載入與串流聊天
└── Tests/ColibriOptionsTests/        swift-testing 測試
```
