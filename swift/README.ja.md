# colibrì for Swift + MLX

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [Italiano](README.it.md) · 日本語

`swift/` は colibrì の Apple Silicon ネイティブなフロントエンドです。
[MLX Swift](https://github.com/ml-explore/mlx-swift) と
[MLX Swift LM](https://github.com/ml-explore/mlx-swift-lm) の上に構築された小さな
Swift 製コマンドラインツールです。MLX は Metal を通じて GPU でモデルを実行し、
ユニファイドメモリを使うため、CPU と GPU の間でデータをコピーしません。

リポジトリには 2 つのエンジンツリーがあります。

| フォルダ | 言語 | 動作環境 | 用途 |
|---|---|---|---|
| [`c/`](../c) | C（+ CUDA、Metal、Vulkan バックエンド） | Linux、Windows、macOS | colibrì 独自のフォーマット、エキスパートキャッシュ、階層メモリを使い、ディスクからストリーミングするフロンティア MoE モデル（744B〜2.8T） |
| `swift/` | Swift + MLX | Apple Silicon 上の macOS 14 以降 | ユニファイドメモリに収まる MLX 形式のモデル（例：`mlx-community/*`） |

Swift ツールは colibrì の量子化 safetensors 形式を読まず、SSD からエキスパートを
ストリーミングしません。RAM に収まらないモデルには C エンジン（`make` と
`coli`、Mac では `METAL=1`）を使ってください。

## 必要環境

- Apple Silicon の Mac、macOS 14 以降
- Xcode 16 以降（Swift 6.2 ツールチェーン）。MLX の Metal シェーダーは
  `xcodebuild` でしかコンパイルできないため、Command Line Tools だけでなく
  Xcode 本体が必要です。

## ビルド

```bash
cd swift
xcodebuild -scheme colibri-swift -destination 'platform=macOS' -derivedDataPath .build/xcode build
```

バイナリは `.build/xcode/Build/Products/Debug/colibri-swift` に出力されます。
最適化ビルドには `-configuration Release` を追加してください（パスの末尾は
`Release/` になります）。

`swift build` でもコードはコンパイルできますが、生成されたバイナリは実行時に
MLX の Metal ライブラリを読み込めません。コンパイル確認のみに使ってください。

## 実行

```bash
# 単一プロンプト（stdout にストリーミング）
.build/xcode/Build/Products/Debug/colibri-swift "mixture-of-experts を 2 文で説明して"

# 対話チャット（空行または Ctrl-D で終了）
.build/xcode/Build/Products/Debug/colibri-swift --model mlx-community/Qwen3-4B-4bit
```

| フラグ | 既定値 | 意味 |
|---|---|---|
| `-m`, `--model` | `mlx-community/Qwen3-4B-4bit` | Hugging Face のモデル ID、または MLX 重みを含むローカルディレクトリ |
| `-n`, `--max-tokens` | `512` | 1 回の応答の最大トークン数 |
| `-t`, `--temp` | `0.6` | サンプリング温度（`0` = greedy） |
| `-s`, `--system` | なし | システムプロンプト |
| `-h`, `--help` | | 使い方を表示 |

ID で指定したモデルは初回使用時に Hugging Face からダウンロードされ、
キャッシュされます。ローカルディレクトリはネットワークなしでディスクから
読み込まれます。Qwen、Llama、Gemma、Mistral、Phi、DeepSeek 系など、
MLX Swift LM が対応するアーキテクチャはすべて使えます。Hugging Face の
チェックポイントを MLX 形式に変換するには、Python の `mlx-lm` パッケージの
`mlx_lm.convert` を使ってください。

## テスト

```bash
cd swift
swift test
```

テストは引数解析を対象とし、MLX に依存しない `ColibriOptions` ターゲットに
あります。

## 構成

```
swift/
├── Package.swift                     SwiftPM マニフェスト（mlx-swift-lm、swift-huggingface、swift-transformers）
├── Sources/ColibriOptions/           コマンドライン解析（MLX 非依存）
├── Sources/ColibriSwift/main.swift   モデル読み込みとストリーミングチャット
└── Tests/ColibriOptionsTests/        swift-testing のテスト
```
