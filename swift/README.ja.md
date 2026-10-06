# Swift 版 colibrì

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [Italiano](README.it.md) · 日本語

`swift/` は Apple Silicon 向けの、単体で完結した colibrì の Swift 実装です。
[`c/`](../c) のものは何も使いません（C コンパイラ、Python ランチャー、`make` は不要）。
2 つのプログラムがあります。

| プログラム | 実行するもの | 方法 |
|---|---|---|
| `coli` | colibrì で変換した MoE モデル（GLM-5.x、DeepSeek-V3 系）。RAM よりはるかに大きいもの | 密な重みは常駐、ルーティングされた expert は SSD から RAM キャッシュへストリーミング。CPU + Accelerate（NEON/AMX） |
| `colibri-swift` | ユニファイドメモリに収まる MLX 形式モデル（`mlx-community/*`） | Metal 経由で GPU 上の [MLX Swift](https://github.com/ml-explore/mlx-swift) |

## `coli` — ストリーミングエンジン

- **コンテナ読み込み。** モデルディレクトリを直接読みます：`config.json`、
  `generation_config.json`、トークナイザー、すべての `*.safetensors` シャードと
  スケール `W.qs`。形式は C エンジンと同じくバイト数から判定します
  （[docs/FORMATS.md](../docs/FORMATS.md)）：f32、int8-row、int4-row、int2-row、
  int4-grouped、int3-g64、fp8-e4m3-b128。
- **SSD からの expert。** attention、埋め込み、正規化、共有 expert は RAM に常駐。
  ルーターが選んだ expert だけを `pread` で読み、1 層分のミスは並列に読み込み、
  `--ram` で決まる LRU キャッシュに保持します。macOS では `F_NOCACHE` を使います。
- **学習するキャッシュ。** expert の使用回数を毎ターン `<model>/.coli_usage` に
  C エンジンと同じ形式で保存し、次回起動時によく使うものを RAM に固定します（`--pin auto`）。
- **モデル。** 圧縮 KV キャッシュ付きの MLA、補正バイアス付きシグモイドルーティング、
  共有 expert、先頭の密な層。KV キャッシュの接頭辞再利用で、各ターンは新しい部分だけを処理します。
- **OpenAI 互換サーバー。** `coli serve` は `/v1/chat/completions`、`/v1/completions`
  （SSE ストリーミング対応）、`/v1/models`、`/health`、`/experts` を提供し、API キー、
  CORS、Host ヘッダー検査、上限付きキューを備えます。

### 要件・ビルド・実行

Apple Silicon Mac（macOS 14 以上）、Xcode 16 以上（Swift 6.2）、SSD 上の変換済みモデル。

```bash
cd swift
swift build -c release --product coli
export COLI_MODEL=/Volumes/ssd/glm-5.2-int4
.build/release/coli info
.build/release/coli run "mixture-of-experts を 2 文で説明して"
.build/release/coli chat --ram 48
.build/release/coli serve --port 8000 --api-key "$KEY"
```

| オプション | 既定値 | 意味 |
|---|---|---|
| `--model DIR` | `$COLI_MODEL` | 変換済みモデルのディレクトリ |
| `--ram GB` | 利用可能メモリの 85% | RAM 予算。残りが expert キャッシュになる |
| `--ctx N` | `8192` | 最大コンテキスト |
| `--pin MODE` | `auto` | `auto`（`.coli_usage` から）、`none`、または履歴ファイル |
| `--pin-gb GB` | キャッシュの半分 | 固定する expert 用の RAM |
| `--topk N` / `--topp P` | モデル値 / 無効 | トークンあたりの expert 数 / 適応選択 |
| `--ngen N`、`--temp T`、`--top-p P`、`--seed N` | 2048、1.0、0.95、ランダム | 生成設定 |
| `--system TEXT`、`--think`、`--no-think` | | システムプロンプト、推論の切り替え |
| `--host`、`--port`、`--model-id`、`--api-key` | 127.0.0.1、8000 | サーバー |
| `--cors-origin`、`--allowed-host`、`--max-queue`、`--web-root` | | セキュリティ、キュー、Web UI |
| `--json` | | `info`/`plan` の機械可読出力 |

API の各応答には `colibri` オブジェクト（tokens/s、expert ヒット率、SSD 読み込みバイト数）が付きます。

### 移植していないもの

Swift エンジンは CPU 上の GLM / DeepSeek-V3（MLA + MoE）系に対応します。他のアーキテクチャ、
`index_topk` を超える DSA 選択（Swift は全トークンに注意を向ける）、MTP、E8/IQ3 形式、GPU カーネル、
複数 SSD ミラー、クラスタ、変換ツール、ベンチマーク、`doctor` は C 版のみです。

## `colibri-swift` — MLX モデル

```bash
cd swift
xcodebuild -scheme colibri-swift -destination 'platform=macOS' -derivedDataPath .build/xcode build
.build/xcode/Build/Products/Debug/colibri-swift --model mlx-community/Qwen3-4B-4bit "こんにちは"
```

## テスト

```bash
cd swift
swift test
```

テストは小さなランダム MoE モデルを書き出して最初から最後まで実行します（逐次デコード＝一括プリフィル、
コールドキャッシュ＝ウォームキャッシュ、KV キャッシュ再利用）。形式、safetensors、`.coli_usage`、
サンプリング、引数、HTTP、OpenAI ルートも検証します。
