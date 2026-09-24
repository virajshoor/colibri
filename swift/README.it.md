# colibrì per Swift + MLX

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · Italiano · [日本語](README.ja.md)

`swift/` è il front end nativo di colibrì per Apple Silicon: un piccolo
strumento a riga di comando in Swift basato su
[MLX Swift](https://github.com/ml-explore/mlx-swift) e
[MLX Swift LM](https://github.com/ml-explore/mlx-swift-lm). MLX esegue il
modello sulla GPU tramite Metal e usa la memoria unificata, quindi nulla viene
copiato tra CPU e GPU.

Il repository ha due alberi di engine:

| Cartella | Linguaggio | Gira su | Usala per |
|---|---|---|---|
| [`c/`](../c) | C (+ backend CUDA, Metal, Vulkan) | Linux, Windows, macOS | Modelli MoE di frontiera (744B–2.8T) in streaming dal disco con i formati, la cache degli esperti e la memoria a livelli di colibrì |
| `swift/` | Swift + MLX | macOS 14+ su Apple Silicon | Modelli pubblicati in formato MLX (ad esempio `mlx-community/*`) che stanno nella memoria unificata |

Lo strumento Swift non legge i formati safetensors quantizzati di colibrì e non fa streaming
degli esperti da SSD. Per i modelli che non stanno in RAM usa l'engine C
(`make` e `coli`, con `METAL=1` su Mac).

## Requisiti

- Mac Apple Silicon, macOS 14 o successivo
- Xcode 16 o successivo (toolchain Swift 6.2). Serve Xcode completo, non solo
  i Command Line Tools, perché MLX include shader Metal che solo `xcodebuild`
  compila.

## Build

```bash
cd swift
xcodebuild -scheme colibri-swift -destination 'platform=macOS' -derivedDataPath .build/xcode build
```

Il binario viene scritto in `.build/xcode/Build/Products/Debug/colibri-swift`.
Aggiungi `-configuration Release` per una build ottimizzata (il percorso
finisce allora in `Release/`).

Anche `swift build` compila il codice, ma il binario risultante non riesce a
caricare la libreria Metal di MLX a runtime. Usalo solo per verificare che il
codice compili.

## Esecuzione

```bash
# Un solo prompt, in streaming su stdout
.build/xcode/Build/Products/Debug/colibri-swift "Spiega i mixture-of-experts in due frasi"

# Chat interattiva (riga vuota o Ctrl-D per uscire)
.build/xcode/Build/Products/Debug/colibri-swift --model mlx-community/Qwen3-4B-4bit
```

| Flag | Default | Significato |
|---|---|---|
| `-m`, `--model` | `mlx-community/Qwen3-4B-4bit` | Id del modello su Hugging Face, o cartella locale con pesi MLX |
| `-n`, `--max-tokens` | `512` | Token massimi per risposta |
| `-t`, `--temp` | `0.6` | Temperatura di campionamento (`0` = greedy) |
| `-s`, `--system` | nessuno | Prompt di sistema |
| `-h`, `--help` | | Mostra l'uso |

I modelli indicati per id vengono scaricati da Hugging Face al primo uso e
messi in cache. Una cartella locale viene caricata dal disco senza rete.
Funziona qualunque architettura supportata da MLX Swift LM, incluse le
famiglie Qwen, Llama, Gemma, Mistral, Phi e DeepSeek. Per convertire un
checkpoint Hugging Face in formato MLX usa `mlx_lm.convert` dal pacchetto
Python `mlx-lm`.

## Test

```bash
cd swift
swift test
```

I test coprono il parsing degli argomenti. Stanno nel target
`ColibriOptions`, che non dipende da MLX.

## Struttura

```
swift/
├── Package.swift                     manifest SwiftPM (mlx-swift-lm, swift-huggingface, swift-transformers)
├── Sources/ColibriOptions/           parsing della riga di comando (senza MLX)
├── Sources/ColibriSwift/main.swift   caricamento del modello e chat in streaming
└── Tests/ColibriOptionsTests/        test swift-testing
```
