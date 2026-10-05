# Nemotron-3-Diarization: conversion and golden fixtures

Tooling for bringing NVIDIA's `Nemotron-3-Diarization` (100 M params, 8 speakers,
10 ms output) into cera as a second, side-by-side diarizer. The 4spk Sortformer port
(`scripts/sortformer/`, `cera::model::sortformer`) is untouched; this directory mirrors
its structure.

| Script | Needs | Does |
|---|---|---|
| `convert.py` | torch, safetensors, numpy, pyyaml, gguf | safetensors + `.nemo` -> GGUF (`f32` / `f16` / `q8_0` / `q4_0`). No NeMo required. |
| `gen_golden.py` | NeMo from the Speech repo (below) | Runs NVIDIA's reference on the clip and writes the golden JSON + tensors. |
| `verify_gguf.py` | NeMo + gguf | Loads a GGUF back into NeMo's modules and compares to the golden, stage by stage. |

Environment (project `.venv`, Python 3.12; what the goldens below were produced with):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch safetensors gguf numpy pyyaml soundfile scipy \
  transformers Cython packaging 'nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech.git@1688cc3d6a9ade854f544987810c53f605dc86fc'
```

PyPI `nemo_toolkit` 3.0.0 cannot load this checkpoint (`self_attention_model='rope'` is
not supported there); the Speech-repo build (3.1.0+1688cc3d6) is required. Pinned
versions otherwise: torch 2.14.1, transformers 5.18.0 (the first release shipping
`models.nemotron3_diarization`, so no git install of transformers is needed).

```bash
M=~/.leap/models/nemotron3-diarization
curl -L -o $M/Nemotron-3-Diarization.nemo \
  https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/main/Nemotron-3-Diarization.nemo
# sha256 867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d
curl -L -o $M/model.safetensors \
  https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/main/model.safetensors
# sha256 c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6
# plus config.json and processor_config.json from the same repo (converter reads config.json)

.venv/bin/python scripts/nemotron3_diarization/convert.py --safetensors $M/model.safetensors \
  --nemo $M/Nemotron-3-Diarization.nemo --out $M/nemotron3-diarization-q8_0.gguf --outtype q8_0
.venv/bin/python scripts/nemotron3_diarization/gen_golden.py --nemo $M/*.nemo \
  --clip cera/tests/fixtures/sortformer/clip.wav \
  --json cera/tests/fixtures/nemotron3/golden.json --tensors $M/golden/golden.safetensors
.venv/bin/python scripts/nemotron3_diarization/verify_gguf.py --nemo $M/*.nemo \
  --gguf $M/nemotron3-diarization-f32.gguf --clip cera/tests/fixtures/sortformer/clip.wav \
  --tensors $M/golden/golden.safetensors --tol-pred 1e-4
cargo test -p cera --test nemotron3_parity   # CERA_REQUIRE_MODEL=1 to fail instead of skip
```

`golden.json` (1.3 MB) is committed. `golden.safetensors` (19 MB, every intermediate
tensor and the per-step cache state) is not; regenerate it with `gen_golden.py`.
Regeneration is reproducible: the trace is asserted equal across two runs.

## What the converter does to the weights

Weights come from `model.safetensors` (+ `config.json` beside it); the `.nemo` is read
only for the mel window/filterbank and the streaming defaults, which `config.json`
does not fully carry. `config.json`'s chunking knobs are the low-latency runtime preset
(fifo 264, update 222), not the checkpoint's own (0, 264) — only the score policy is
cross-checked between the two sources.

* `nd.embed.proj`: the feature-stacking projection, `[512, 1024]`, no bias.
* The 31 encoder layers go in as separate q/k/v (NeMo fuses them into `w_qkv`; the
  verifier re-fuses as `[q; k; v]`). q/k/v have no bias; `attn_o` does.
* The subpixel Conv1d `[1536, 192, 3]` is stored 2D `[1536, 576]`, row-major: the Rust
  port lowers the convolution to a matmul over unrolled frames
  (`unrolled[t, c*3+d]` = mel-frame `t+d-1` of channel `c`, zero outside).
* `--outtype` quantizes the embedder projection and the encoder matrices. Norms, biases,
  silence embeds, mel tables, `nd.proj`, the upsampler and the classifier stay F32 (f32)
  or F16 (otherwise).
* `--tail-outtype q8_0` also quantizes `nd.proj`, the upsampler and the classifier to
  Q8_0. The Hexagon NPU's matmul reads only Q8_0 or Q4_0, so the NPU port needs this
  variant.

## Results on the committed clip (NeMo as the executor)

`verify_gguf.py`, offline pass, 12416 speaker decisions (1552 frames x 8; the mel is
padded to a multiple of 16):

| GGUF | size | worst stage cosine | sigmoid max abs diff | decisions flipped at 0.5 |
|---|---|---|---|---|
| f32 | 397 MB | 1.0000000 | 0.0 | 0 |
| f16 | 199 MB | 1.0000000 | 1.2e-6 | 0 |
| q8_0 | 107 MB | 0.9998252 | 3.2e-2 | 0 |
| q4_0 | 58 MB | 0.9556320 | 4.1e-1 | 78 |

The f32 row is bit-exact, which proves the HuggingFace safetensors match the `.nemo`
bit-for-bit through the whole network. One 15 s clip and a flip count are not a
diarization error rate. Q8_0 looks free; Q4_0 flips 78 decisions, so do not ship it
before a DER run on real meeting data. (These use gguf-py's reference quantizer; cera's
own may round differently.)

The erf-vs-tanh GELU spike (`gelu_spike.py`, Transformers executor): tanh
substitution through full low-latency streaming gives max-abs 7.2e-3 with 0 flips of
12296, so the NPU (which has no erf GELU) uses the tanh form.

## What the golden captures

* Offline: mel, stacked embeddings, input-LN output, all 31 block outputs, final-LN
  output, `encoder_proj`, the upsampled classifier input, the logits, the sigmoids
  (and the auxiliary activity logits, which the GGUF intentionally omits).
* Streaming, five presets: the checkpoint default (264/0/0, no FIFO), the model card's
  low-latency (9/4, FIFO 264) and ultra-low-latency (3/1) presets, and two tiny-cache
  presets (`tiny`, `tiny_nofifo`) that make a 15 s clip overflow the speaker cache so
  compression, learned silence slots and FIFO pop are exercised. Per step: frame bounds,
  chunk predictions, and the state after the update (cache, cache lengths, cache
  predictions, compression flag, FIFO, FIFO lengths, FIFO predictions, speaker
  permutation, mean silence embedding, silence count). The unrolled loop is asserted
  equal to NeMo's own `forward_streaming`, and two runs are asserted identical.
* The generator asserts the mel front-end defaults the converter hard-codes (pre-emphasis
  0.97, log guard 2^-24, power 2, pad_to 16, normalize NA) and that there is no
  transformer head and the silence embedding is learned. Dither (1e-5) is training-only
  in NeMo (`features.py`: `self.dither if self.training else 0.0`), confirmed in eval by
  two identical `diarize` runs.

Facts a port must match, all read from NeMo 3.1.0+1688cc3d (Speech repo) and the
Transformers 5.18.0 port rather than assumed:

* RoPE positions restart at 0 for every chunk over `[cache, fifo, chunk, lookahead]`;
  standard NeoX rotation, full head dim, theta 10000.
* The MLP activation is erf GELU on both sides (`GELU(approximate='none')`,
  `GELUActivation`); q/k/v have no bias, `out_proj` does; the encoder is pre-LN with
  input and final norms.
* The cache scores pooled (sigmoid-then-average x8) probabilities; the step runs the
  encoder over `[cache, fifo, chunk, lookahead]`, then proj, subpixel upsample
  (Conv1d k=3 pad 1 + interleave), ReLU -> Linear -> ReLU -> Linear(->8); chunk
  predictions exclude the lookahead, which never joins the FIFO; the last chunk has no
  lookahead.
* There is no left context (checkpoint default 0, and the Transformers port has none).
* Mel is computed over the whole clip and then sliced into chunks (`pad_to=16` pads the
  feature length); an incremental mel front end has to reproduce the clip-level result
  at chunk edges.
* The default `async_streaming=False` path is what is captured.

## The Rust model

`cera::model::nemotron3_diarization` loads these GGUFs and runs the whole pipeline on
the CPU, reusing the checkpoint's own window and filterbank. `cera/tests/nemotron3_parity.rs`
pins it to NeMo; `cera/src/model/nemotron3_diarization_hexagon.rs` stages the network
on the Hexagon NPU behind the same accelerator split as the 4spk port (mel, embedder,
whole-network predict on the DSP; cache update on the CPU).
