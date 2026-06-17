# cera-cli

Command-line interface for the [`cera`](https://github.com/hyeons-lab/cera/tree/main/cera) LLM inference engine. Installs
a `cera` binary for running, chatting with, inspecting, and benchmarking GGUF /
LeapBundles models locally.

> **Note:** Part of a learning-experiment project exploring LLM inference
> internals in Rust — see the [project README](https://github.com/hyeons-lab/cera).
> Not intended for production use.

## Install

```sh
cargo install cera-cli
```

This builds the `cera` binary. For an Apple Metal or wgpu GPU build:

```sh
cargo install cera-cli --features metal   # or gpu
```

## Usage

Point at a local model — a `.gguf` file, a `.json` LeapBundles manifest, or a
directory containing one — or let it auto-download a bundle by id/quant from
[`huggingface.co/LiquidAI/LeapBundles`](https://huggingface.co/LiquidAI/LeapBundles)
(cached under `$HOME/.cache/cera`). Supported architectures: `lfm2`,
`qwen2`/`qwen3`, `llama` (incl. classic Mistral), and `granite` — see the
[`cera` README](https://github.com/hyeons-lab/cera/tree/main/cera#supported-models)
for the full list and modality support.

```sh
# Generate from a local GGUF
cera run --model model.gguf --prompt "Explain quantization in one sentence."

# Auto-download a bundle and generate
cera run --bundle-id LFM2.5-1.2B-Instruct --quant Q4_0 --prompt "Hello"

# Constrain output to valid JSON (bundled grammar) or a custom GBNF
cera run -m model.gguf -p "List 3 colors as JSON" --json
cera run -m model.gguf -p "..." --grammar @schema.gbnf

# Interactive multi-turn chat REPL (keeps the prefix cache warm across turns)
cera chat --bundle-id LFM2.5-1.2B-Instruct --quant Q4_0
```

### Commands

| Command | Purpose |
|---------|---------|
| `run` | Run inference on a prompt — text, optional grammar/JSON, plus audio input for LFM2-Audio bundles. |
| `chat` | Interactive multi-turn REPL with `/help`, `/clear`, `/exit` slash commands. |
| `inspect` | Inspect a GGUF file's metadata and resolved CPU backend tier. |
| `cpu` | Print the host's CPU backend tier + detected SIMD features (no model needed). |
| `tokenize` | Tokenize text and print token IDs (e.g. to compare against HuggingFace). |
| `bench` | Measure decode throughput (tok/s) with p10/p50/p90/mean/stddev over N runs. |
| `list-bundles` | List bundles on `LiquidAI/LeapBundles` (add `--quants` for per-bundle quants). |
| `download-bundles` | Download bundle manifests + model files without loading them. |

Run `cera <command> --help` for the full flag list. Common `run` flags:
`--max-tokens` (default 256), `--temperature` (default 0.7), `--device`
(`cpu` / `gpu` / `auto`, default `auto`), `--grammar` / `--json`.

## License

Apache-2.0 OR MIT.
