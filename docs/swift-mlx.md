# Swift engine (Apple Silicon)

The repository has two independent implementations. `c/` is the C engine for Linux,
Windows and macOS with CUDA, Vulkan and [Metal](metal.md) backends. `swift/` is a
self-contained Swift implementation for Apple Silicon that needs nothing from `c/`:

- **`coli`** (Swift) runs colibrì's converted MoE models (GLM-5.x, DeepSeek-V3
  family) that are far larger than RAM. It reads the quantized safetensors
  containers directly (f32, int8-row, int4-row, int2-row, int4-grouped, int3-g64,
  fp8-e4m3-b128; see [FORMATS](FORMATS.md)), keeps the dense weights in RAM, streams
  routed experts from SSD into an LRU cache, pins the most used experts from
  `.coli_usage`, and serves an OpenAI-compatible API.
- **`colibri-swift`** runs MLX-format models (`mlx-community/*`) on the GPU with
  [MLX Swift](https://github.com/ml-explore/mlx-swift).

## Build and run `coli`

Requires an Apple Silicon Mac on macOS 14+ and Xcode 16+.

```bash
cd swift
swift build -c release --product coli
.build/release/coli info --model /Volumes/ssd/glm-5.2-int4
.build/release/coli chat --model /Volumes/ssd/glm-5.2-int4 --ram 48
.build/release/coli serve --model /Volumes/ssd/glm-5.2-int4 --port 8000
```

| Command | Does |
|---|---|
| `run "prompt"` | One-shot generation streamed to stdout |
| `chat` | Interactive chat with KV-cache reuse across turns |
| `serve` | `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/experts` |
| `info`, `plan` | Model, formats and memory plan, without loading weights (`--json` available) |

Main options: `--ram GB` (RAM budget, default 85% of available), `--ctx N`,
`--pin auto|none|PATH`, `--topk N` / `--topp P` (experts per token), `--ngen`,
`--temp`, `--top-p`, `--seed`, `--think` / `--no-think`, and for `serve`
`--host`, `--port`, `--api-key`, `--cors-origin`, `--allowed-host`,
`--max-queue`, `--web-root`. Responses include a `colibri` statistics object
(tokens/s, expert hit rate, SSD bytes read).

## Build and run `colibri-swift`

MLX's Metal shaders are only compiled by `xcodebuild`:

```bash
cd swift
xcodebuild -scheme colibri-swift -destination 'platform=macOS' -derivedDataPath .build/xcode build
.build/xcode/Build/Products/Debug/colibri-swift --model mlx-community/Qwen3-4B-4bit "Hello"
```

## Scope

The Swift engine implements the GLM / DeepSeek-V3 MLA + MoE family on the CPU with
Accelerate. Other architectures, the DSA selection beyond `index_topk` tokens,
multi-token prediction, the E8/IQ3 format, GPU kernels, multi-SSD mirroring,
clustering and the converter remain C-only. Run `swift test` in `swift/` for the
test suites. Full details are in
[`swift/README.md`](https://github.com/JustVugg/colibri/blob/main/swift/README.md),
also available in 简体中文, 繁體中文, Italiano and 日本語.
