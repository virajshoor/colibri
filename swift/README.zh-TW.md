# Swift 版 colibrì

[English](README.md) · [简体中文](README.zh-CN.md) · 繁體中文 · [Italiano](README.it.md) · [日本語](README.ja.md)

`swift/` 是針對 Apple Silicon 的完整、獨立的 colibrì Swift 實作，不依賴
[`c/`](../c) 中的任何東西（不需要 C 編譯器、Python 啟動器或 `make`）。包含兩個程式：

| 程式 | 執行什麼 | 方式 |
|---|---|---|
| `coli` | colibrì 轉換的 MoE 模型（GLM-5.x、DeepSeek-V3 系列），遠大於記憶體 | 稠密權重常駐記憶體，路由專家從 SSD 串流讀入記憶體快取；CPU + Accelerate（NEON/AMX） |
| `colibri-swift` | 能放進統一記憶體的 MLX 格式模型（`mlx-community/*`） | 透過 Metal 在 GPU 上執行 [MLX Swift](https://github.com/ml-explore/mlx-swift) |

## `coli` — 串流引擎

- **容器讀取。** 直接讀取模型目錄：`config.json`、`generation_config.json`、分詞器檔案
  與所有 `*.safetensors` 分片，包括 `W.qs` 縮放檔。格式與 C 引擎一樣由位元組數判定
  （[docs/FORMATS.md](../docs/FORMATS.md)）：f32、int8-row、int4-row、int2-row、
  int4-grouped、int3-g64、fp8-e4m3-b128。
- **從 SSD 讀取專家。** 注意力、嵌入、正規化與共享專家常駐記憶體。路由器選中的專家才用
  `pread` 讀取，一層的所有未命中平行讀取，並保存在由 `--ram` 決定大小的 LRU 快取中。
  macOS 上使用 `F_NOCACHE`。
- **會學習的快取。** 每輪把專家使用次數寫入 `<模型>/.coli_usage`（與 C 引擎格式相同），
  下次啟動時把最常用的專家固定在記憶體中（`--pin auto`）。
- **模型。** 帶壓縮 KV 快取的 MLA、帶校正偏置的 sigmoid 路由、共享專家與前幾層稠密層。
  KV 快取前綴重用讓每輪只預填新內容。
- **OpenAI 相容伺服器。** `coli serve` 提供 `/v1/chat/completions`、`/v1/completions`
  （支援 SSE 串流）、`/v1/models`、`/health`、`/experts`，支援 API 金鑰、CORS、
  Host 標頭檢查與有限佇列。

### 需求、建置、執行

Apple Silicon Mac（macOS 14+）、Xcode 16+（Swift 6.2）、放在 SSD 上的已轉換模型。

```bash
cd swift
swift build -c release --product coli
export COLI_MODEL=/Volumes/ssd/glm-5.2-int4
.build/release/coli info
.build/release/coli run "用兩句話解釋混合專家模型"
.build/release/coli chat --ram 48
.build/release/coli serve --port 8000 --api-key "$KEY"
```

| 選項 | 預設值 | 含義 |
|---|---|---|
| `--model DIR` | `$COLI_MODEL` | 已轉換模型目錄 |
| `--ram GB` | 可用記憶體的 85% | 記憶體預算，剩餘部分給專家快取 |
| `--ctx N` | `8192` | 最大上下文 |
| `--pin MODE` | `auto` | `auto`（來自 `.coli_usage`）、`none` 或歷史檔路徑 |
| `--pin-gb GB` | 快取的一半 | 固定專家使用的記憶體 |
| `--topk N` / `--topp P` | 模型值 / 關閉 | 每個 token 的專家數 / 自適應選擇 |
| `--ngen N`、`--temp T`、`--top-p P`、`--seed N` | 2048、1.0、0.95、隨機 | 生成參數 |
| `--system TEXT`、`--think`、`--no-think` | | 系統提示、推理開關 |
| `--host`、`--port`、`--model-id`、`--api-key` | 127.0.0.1、8000 | 伺服器 |
| `--cors-origin`、`--allowed-host`、`--max-queue`、`--web-root` | | 安全、佇列、網頁介面 |
| `--json` | | `info`/`plan` 的機器可讀輸出 |

每個 API 回應都帶有 `colibri` 物件（tokens/s、專家命中率、從 SSD 讀取的位元組數）。

### 尚未移植的部分

Swift 引擎在 CPU 上支援 GLM / DeepSeek-V3（MLA + MoE）系列。其他架構、超過 `index_topk`
個 token 後的 DSA 選擇（Swift 會關注全部 token）、MTP、E8/IQ3 格式、GPU 核心、多 SSD 鏡像、
叢集、轉換工具、基準測試與 `doctor` 仍只在 C 版中。

## `colibri-swift` — MLX 模型

```bash
cd swift
xcodebuild -scheme colibri-swift -destination 'platform=macOS' -derivedDataPath .build/xcode build
.build/xcode/Build/Products/Debug/colibri-swift --model mlx-community/Qwen3-4B-4bit "你好"
```

## 測試

```bash
cd swift
swift test
```

測試會寫出一個隨機的小型 MoE 模型並端到端執行（逐步解碼等於整體預填、冷快取等於熱快取、
KV 快取重用），並涵蓋各量化格式、safetensors、`.coli_usage`、取樣、參數解析、HTTP 與 OpenAI 路由。
