# Swift 版 colibrì

[English](README.md) · 简体中文 · [繁體中文](README.zh-TW.md) · [Italiano](README.it.md) · [日本語](README.ja.md)

`swift/` 是面向 Apple Silicon 的完整、独立的 colibrì Swift 实现，不依赖
[`c/`](../c) 中的任何东西（无需 C 编译器、Python 启动器或 `make`）。包含两个程序：

| 程序 | 运行什么 | 方式 |
|---|---|---|
| `coli` | colibrì 转换的 MoE 模型（GLM-5.x、DeepSeek-V3 系列），远大于内存 | 稠密权重常驻内存，路由专家从 SSD 流式读入内存缓存；CPU + Accelerate（NEON/AMX） |
| `colibri-swift` | 能放进统一内存的 MLX 格式模型（`mlx-community/*`） | 通过 Metal 在 GPU 上运行 [MLX Swift](https://github.com/ml-explore/mlx-swift) |

## `coli` — 流式引擎

- **容器读取。** 直接读取模型目录：`config.json`、`generation_config.json`、分词器文件
  和所有 `*.safetensors` 分片，包括 `W.qs` 缩放文件。格式与 C 引擎一样由字节数判定
  （[docs/FORMATS.md](../docs/FORMATS.md)）：f32、int8-row、int4-row、int2-row、
  int4-grouped、int3-g64、fp8-e4m3-b128。
- **从 SSD 读取专家。** 注意力、嵌入、归一化和共享专家常驻内存。路由器选中的专家才用
  `pread` 读取，一层的所有未命中并行读取，并保存在由 `--ram` 决定大小的 LRU 缓存中。
  macOS 上使用 `F_NOCACHE`。
- **会学习的缓存。** 每轮把专家使用次数写入 `<模型>/.coli_usage`（与 C 引擎格式相同），
  下次启动时把最常用的专家固定在内存中（`--pin auto`）。
- **模型。** 带压缩 KV 缓存的 MLA、带校正偏置的 sigmoid 路由、共享专家和前几层稠密层。
  KV 缓存前缀复用使每轮只预填充新内容。
- **OpenAI 兼容服务器。** `coli serve` 提供 `/v1/chat/completions`、`/v1/completions`
  （支持 SSE 流式）、`/v1/models`、`/health`、`/experts`，支持 API 密钥、CORS、
  Host 头检查和有限队列。

### 要求、构建、运行

Apple Silicon Mac（macOS 14+）、Xcode 16+（Swift 6.2）、放在 SSD 上的已转换模型。

```bash
cd swift
swift build -c release --product coli
export COLI_MODEL=/Volumes/ssd/glm-5.2-int4
.build/release/coli info
.build/release/coli run "用两句话解释混合专家模型"
.build/release/coli chat --ram 48
.build/release/coli serve --port 8000 --api-key "$KEY"
```

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--model DIR` | `$COLI_MODEL` | 已转换模型目录 |
| `--ram GB` | 可用内存的 85% | 内存预算，剩余部分给专家缓存 |
| `--ctx N` | `8192` | 最大上下文 |
| `--pin MODE` | `auto` | `auto`（来自 `.coli_usage`）、`none` 或历史文件路径 |
| `--pin-gb GB` | 缓存的一半 | 固定专家使用的内存 |
| `--topk N` / `--topp P` | 模型值 / 关闭 | 每个 token 的专家数 / 自适应选择 |
| `--ngen N`、`--temp T`、`--top-p P`、`--seed N` | 2048、1.0、0.95、随机 | 生成参数 |
| `--system TEXT`、`--think`、`--no-think` | | 系统提示、推理开关 |
| `--host`、`--port`、`--model-id`、`--api-key` | 127.0.0.1、8000 | 服务器 |
| `--cors-origin`、`--allowed-host`、`--max-queue`、`--web-root` | | 安全、队列、网页界面 |
| `--json` | | `info`/`plan` 的机器可读输出 |

每个 API 响应都带有 `colibri` 对象（tokens/s、专家命中率、从 SSD 读取的字节数）。

### 尚未移植的部分

Swift 引擎在 CPU 上支持 GLM / DeepSeek-V3（MLA + MoE）系列。其他架构、超过 `index_topk`
个 token 后的 DSA 选择（Swift 会关注全部 token）、MTP、E8/IQ3 格式、GPU 内核、多 SSD 镜像、
集群、转换工具、基准测试和 `doctor` 仍只在 C 版中。

## `colibri-swift` — MLX 模型

```bash
cd swift
xcodebuild -scheme colibri-swift -destination 'platform=macOS' -derivedDataPath .build/xcode build
.build/xcode/Build/Products/Debug/colibri-swift --model mlx-community/Qwen3-4B-4bit "你好"
```

## 测试

```bash
cd swift
swift test
```

测试会写出一个随机的小型 MoE 模型并端到端运行（逐步解码等于整体预填充、冷缓存等于热缓存、
KV 缓存复用），并覆盖各量化格式、safetensors、`.coli_usage`、采样、参数解析、HTTP 和 OpenAI 路由。
