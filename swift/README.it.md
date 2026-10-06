# colibrì in Swift

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · Italiano · [日本語](README.ja.md)

`swift/` è un'implementazione Swift completa e autonoma di colibrì per Apple Silicon.
Non usa nulla di [`c/`](../c): niente compilatore C, niente launcher Python, niente
`make`. Contiene due programmi:

| Programma | Cosa esegue | Come |
|---|---|---|
| `coli` | i modelli MoE convertiti da colibrì (GLM-5.x, famiglia DeepSeek-V3), molto più grandi della RAM | pesi densi residenti, expert instradati letti da SSD in una cache in RAM; CPU + Accelerate (NEON/AMX) |
| `colibri-swift` | modelli in formato MLX (`mlx-community/*`) che stanno in memoria unificata | [MLX Swift](https://github.com/ml-explore/mlx-swift) sulla GPU via Metal |

## `coli` — il motore in streaming

- **Lettore del container.** Legge direttamente la cartella del modello:
  `config.json`, `generation_config.json`, il tokenizer e tutti gli shard
  `*.safetensors`, comprese le scale `W.qs`. I formati si riconoscono dai byte
  come nel motore C ([docs/FORMATS.md](../docs/FORMATS.md)): f32, int8-row,
  int4-row, int2-row, int4-grouped, int3-g64, fp8-e4m3-b128.
- **Expert da SSD.** Attention, embedding, norme ed expert condivisi restano in RAM.
  Gli expert instradati si leggono con `pread` solo quando il router li sceglie, tutti
  i miss di un layer in parallelo, e restano in una cache LRU dimensionata da
  `--ram`. Su macOS le letture usano `F_NOCACHE`.
- **Cache che impara.** L'uso degli expert si salva in `<modello>/.coli_usage` a ogni
  turno, nello stesso formato del motore C; all'avvio i più usati vengono fissati in
  RAM (`--pin auto`).
- **Modello.** Multi-head latent attention con KV cache compressa, routing sigmoide
  con bias di correzione, expert condivisi e primi layer densi. Il riuso del prefisso
  della KV cache fa elaborare a ogni turno solo il testo nuovo.
- **Server compatibile OpenAI.** `coli serve` espone `/v1/chat/completions` e
  `/v1/completions` (con o senza SSE), `/v1/models`, `/health` e `/experts`, con API
  key, CORS, controllo dell'header Host e coda limitata.

### Requisiti, build, uso

Mac Apple Silicon con macOS 14+, Xcode 16+ (Swift 6.2), un modello convertito su SSD.

```bash
cd swift
swift build -c release --product coli
export COLI_MODEL=/Volumes/ssd/glm-5.2-int4
.build/release/coli info
.build/release/coli run "Spiega i mixture-of-experts in due frasi"
.build/release/coli chat --ram 48
.build/release/coli serve --port 8000 --api-key "$KEY"
```

| Opzione | Default | Significato |
|---|---|---|
| `--model DIR` | `$COLI_MODEL` | cartella del modello convertito |
| `--ram GB` | 85% della RAM disponibile | budget RAM; la cache expert prende il resto |
| `--ctx N` | `8192` | contesto massimo |
| `--pin MODE` | `auto` | `auto` (da `.coli_usage`), `none` o un file di storia |
| `--pin-gb GB` | metà cache | RAM per gli expert fissati |
| `--topk N` / `--topp P` | del modello / spento | expert per token / selezione adattiva |
| `--ngen N`, `--temp T`, `--top-p P`, `--seed N` | 2048, 1.0, 0.95, casuale | generazione |
| `--system TEXT`, `--think`, `--no-think` | | prompt di sistema, ragionamento |
| `--host`, `--port`, `--model-id`, `--api-key` | 127.0.0.1, 8000 | server |
| `--cors-origin`, `--allowed-host`, `--max-queue`, `--web-root` | | sicurezza, coda, interfaccia web |
| `--json` | | output macchina per `info`/`plan` |

Ogni risposta API contiene un oggetto `colibri` (token/s, hit rate degli expert,
byte letti da SSD).

### Cosa non è portato

Il motore Swift copre la famiglia GLM / DeepSeek-V3 (MLA + MoE) su CPU. Restano solo
in C: le altre architetture, la selezione DSA oltre `index_topk` token (Swift
considera tutti i token), MTP, il formato E8/IQ3, i kernel GPU, il mirroring
multi-SSD, il cluster, il convertitore, benchmark e `doctor`.

## `colibri-swift` — modelli MLX

```bash
cd swift
xcodebuild -scheme colibri-swift -destination 'platform=macOS' -derivedDataPath .build/xcode build
.build/xcode/Build/Products/Debug/colibri-swift --model mlx-community/Qwen3-4B-4bit "Ciao"
```

## Test

```bash
cd swift
swift test
```

I test scrivono un piccolo modello MoE casuale e lo eseguono da capo a fondo
(decodifica incrementale = prefill completo, cache fredda = cache calda, riuso della
KV cache), oltre a formati, safetensors, `.coli_usage`, sampling, argomenti, HTTP e
rotte OpenAI.
