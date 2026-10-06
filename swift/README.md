# colibrì in Swift

English · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [Italiano](README.it.md) · [日本語](README.ja.md)

`swift/` is a complete, self-contained Swift implementation of colibrì for Apple
Silicon. It needs nothing from [`c/`](../c): no C compiler, no Python launcher, no
`make`. It contains two programs:

| Program | What it runs | How |
|---|---|---|
| `coli` | colibrì's converted MoE models (GLM-5.x, DeepSeek-V3 family) far larger than RAM | Dense weights resident, routed experts streamed from SSD into a RAM cache; CPU + Accelerate (NEON/AMX) |
| `colibri-swift` | MLX-format models (`mlx-community/*`) that fit in unified memory | [MLX Swift](https://github.com/ml-explore/mlx-swift) on the GPU through Metal |

## `coli` — the streaming engine

`coli` is a Swift port of colibrì's engine, launcher and OpenAI server:

- **Container reader.** Reads the model directory directly: `config.json`,
  `generation_config.json`, the tokenizer files and every `*.safetensors` shard,
  including the `W.qs` scale sidecars of colibrì's quantized formats. Formats are
  identified from byte counts exactly like the C engine
  ([docs/FORMATS.md](../docs/FORMATS.md)): f32, int8-row, int4-row, int2-row,
  int4-grouped, int3-g64 and fp8-e4m3-b128. BF16/F16/F32 tensors are read as floats.
- **Experts from SSD.** Attention, embeddings, norms and shared experts stay in RAM.
  Routed experts are read with `pread` only when the router picks them, all misses of
  a layer in parallel to keep the SSD queue deep, and kept in an LRU cache sized from
  `--ram`. On macOS reads use `F_NOCACHE` so experts are not cached twice.
- **Learning cache.** Expert usage is written to `<model>/.coli_usage` after every
  turn, in the same format as the C engine, and the most used experts are pinned in
  RAM at the next start (`--pin auto`).
- **Model.** Multi-head latent attention with the compressed KV cache (latent +
  shared rotary key per token), sigmoid routing with correction bias, `norm_topk_prob`
  and `routed_scaling_factor`, shared experts and dense first layers. KV-cache prefix
  reuse makes each chat turn prefill only the new text.
- **OpenAI-compatible server.** `coli serve` exposes `/v1/chat/completions` and
  `/v1/completions` (streaming over SSE or not), `/v1/models`, `/health` and
  `/experts`, with API keys, CORS, a DNS-rebinding Host check and a bounded queue.

### Requirements

- Apple Silicon Mac, macOS 14 or newer
- Xcode 16 or newer (Swift 6.2 toolchain)
- A colibrì-converted model directory on a fast SSD

### Build

```bash
cd swift
swift build -c release --product coli
```

The binary is `.build/release/coli`. `coli` has no Metal shaders, so plain
`swift build` is enough.

### Run

```bash
export COLI_MODEL=/Volumes/ssd/glm-5.2-int4    # or pass --model
.build/release/coli info                        # model, formats, memory plan
.build/release/coli run "Explain mixture-of-experts in two sentences"
.build/release/coli chat --ram 48
.build/release/coli serve --port 8000 --api-key "$KEY"
```

Then point any OpenAI client at `http://127.0.0.1:8000/v1`:

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

| Option | Default | Meaning |
|---|---|---|
| `--model DIR` | `$COLI_MODEL` | Converted model directory |
| `--ram GB` | 85% of available | RAM budget; the expert cache gets what resident weights and KV leave |
| `--ctx N` | `8192` | Maximum context (prompt + reply) |
| `--pin MODE` | `auto` | `auto` pins from `.coli_usage`, `none`, or a history file path |
| `--pin-gb GB` | half the cache | RAM given to pinned experts |
| `--topk N` | model's | Experts per token (fewer = less disk traffic, lower quality) |
| `--topp P` | off | Adaptive experts: smallest set reaching this routing mass |
| `--ngen N` | `2048` | Maximum reply tokens |
| `--temp T` | `1.0` | Temperature (`0` = greedy) |
| `--top-p P` | `0.95` | Nucleus sampling |
| `--seed N` | random | Reproducible sampling |
| `--system TEXT` | none | System prompt |
| `--think` / `--no-think` | template default | Toggle reasoning in templates that support it |
| `--host`, `--port` | `127.0.0.1`, `8000` | Server address |
| `--model-id ID` | directory name | Model name reported by the API |
| `--api-key KEY` | `$COLI_API_KEY` | Require `Authorization: Bearer KEY` on `/v1/*` and `/experts` |
| `--cors-origin O` | none | Allowed browser origin (repeatable, `*` for any) |
| `--allowed-host H` | localhost names | Accepted `Host` headers (repeatable) |
| `--max-queue N` | `8` | Requests allowed to wait; more get HTTP 503 |
| `--web-root DIR` | none | Serve a web UI (e.g. `web/dist`) from `/` |
| `--json` | off | Machine-readable `info` / `plan` |

Environment: `COLI_THREADS` sets the worker thread count (default: performance
cores), `COLI_PAGE_CACHE=1` lets macOS cache expert reads.

Every API response carries a `colibri` object (tokens/s, prefill time, expert hit
rate, bytes read from SSD); streams send it as an extra event before `[DONE]`.
Request fields understood: `messages` (text content, string or parts), `prompt`,
`max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `stop`,
`seed`, `stream`, `stream_options.include_usage`,
`chat_template_kwargs.enable_thinking` and `reasoning_effort`.

### What is not ported

The Swift engine covers the GLM / DeepSeek-V3 MLA + MoE family on the CPU. These
parts of the C tree have no Swift counterpart yet: the other architectures
(Qwen 3.6/3.8, Kimi K3, DeepSeek-V4, Inkling, OLMoE), the DSA indexer's selection
beyond `index_topk` tokens (Swift attends to every token instead), multi-token
prediction, the E8/IQ3 lattice format, CUDA/Vulkan/Metal kernels, multi-SSD
mirroring, clustering, the weight converter, benchmarks and `doctor`. Convert
models with the existing tooling, then run them with `coli`.

## `colibri-swift` — MLX models

```bash
cd swift
xcodebuild -scheme colibri-swift -destination 'platform=macOS' -derivedDataPath .build/xcode build
.build/xcode/Build/Products/Debug/colibri-swift --model mlx-community/Qwen3-4B-4bit "Hello"
```

MLX ships Metal shaders that only `xcodebuild` compiles, so build this one with
Xcode. Flags: `-m/--model` (Hugging Face id or local MLX directory, default
`mlx-community/Qwen3-4B-4bit`), `-n/--max-tokens` (512), `-t/--temp` (0.6),
`-s/--system`. With no prompt it starts an interactive chat.

## Test

```bash
cd swift
swift test
```

The tests write a tiny random MoE model to a temporary directory and run it end to
end: incremental decoding must match a full prefill, a cold (streaming-only) expert
cache must match a warm one, and the engine must reuse its KV cache across turns.
They also cover every quantized format, the safetensors reader, the `.coli_usage`
format, sampling, argument parsing, the HTTP parser and the OpenAI routes.

## Layout

```
swift/
├── Package.swift
├── Sources/ColibriCore/       engine: safetensors, formats, kernels, expert cache, model, sampling, generation
├── Sources/ColibriServer/     HTTP/1.1 server and OpenAI routes
├── Sources/ColiCLI/           the `coli` command and the tokenizer adapter (swift-transformers)
├── Sources/ColibriOptions/    `colibri-swift` argument parsing
├── Sources/ColibriSwift/      `colibri-swift` (MLX)
└── Tests/                     swift-testing suites
```
