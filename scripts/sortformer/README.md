# Streaming Sortformer: conversion and golden fixtures

Tooling for bringing NVIDIA's `diar_streaming_sortformer_4spk-v2.1` (117 M params, 4 speakers) into
cera. The design and the NPU plan are in the repository's `.agents/plans/` directory (`background-diarization-sortformer`).

| Script | Needs | Does |
|---|---|---|
| `convert_sortformer.py` | torch, numpy, pyyaml, gguf | `.nemo` -> GGUF (`f32` / `f16` / `q8_0` / `q4_0`). No NeMo required. |
| `make_clip.py` | soundfile, scipy | Rebuilds the 15 s multi-voice test clip (committed; only needed to regenerate it). |
| `gen_golden.py` | NeMo (`nemo_toolkit[asr]`) | Runs NVIDIA's reference on the clip and writes the golden JSON + tensors. |
| `verify_gguf.py` | NeMo + gguf | Loads a GGUF back into NeMo's modules and compares to the golden, stage by stage. |

```bash
M=~/.leap/models/sortformer
curl -L -o $M/diar_streaming_sortformer_4spk-v2.1.nemo \
  https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1/resolve/main/diar_streaming_sortformer_4spk-v2.1.nemo
# sha256 8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8

python scripts/sortformer/convert_sortformer.py --nemo $M/*.nemo --out $M/sortformer-4spk-v2.1-q8_0.gguf --outtype q8_0
python scripts/sortformer/gen_golden.py --nemo $M/*.nemo --clip cera/tests/fixtures/sortformer/clip.wav \
  --json cera/tests/fixtures/sortformer/golden.json --tensors $M/golden/golden.safetensors
python scripts/sortformer/verify_gguf.py --nemo $M/*.nemo --gguf $M/sortformer-4spk-v2.1-f32.gguf \
  --clip cera/tests/fixtures/sortformer/clip.wav --tensors $M/golden/golden.safetensors --tol-pred 1e-4
cargo test -p cera --test sortformer_gguf_layout   # CERA_REQUIRE_MODEL=1 to fail instead of skip
cargo test -p cera --release --test sortformer_parity   # the CPU model against NeMo; add -- --include-ignored for the 33-step preset
```

`golden.json` and `clip.wav` are committed (about 650 KB). `golden.safetensors` (15 MB, every
intermediate tensor and the per-step cache state) is not; regenerate it with `gen_golden.py`.
Regeneration is reproducible: two runs give identical JSON.

## What the converter does to the weights

* FastConformer tensors keep the LFM2-Audio mmproj names and shapes (`a.conv1d.*`,
  `a.pre_encode.*`, `a.blk.N.*`), so the existing encoder block loader reads them. The metadata key
  `clip.audio.feed_forward_length` is written with the true value (2048); the mmproj's says 512 and
  cera works around it by reading the tensor.
* `conv.batch_norm` is folded into the per-channel `conv_norm` affine (`w / sqrt(var + 1e-5)`,
  `b - mean * scale`, in f64).
* The singleton axis of the pointwise and depthwise conv weights is dropped.
* Transformer head, `encoder_proj`, the speaker head and the mel window/filterbank go in under
  `sf.*`. NeMo's `hidden_to_spks` is not exported (unused by `forward_speaker_sigmoids`).
* `--outtype` quantizes only the FastConformer matrices. Everything on the path to the sigmoids
  (head, `encoder_proj`) is F16 at most.

## Results on the committed clip (NeMo as the executor)

`verify_gguf.py`, offline pass, 776 speaker decisions (194 frames x 4; 193 are speech frames, one is padding):

| GGUF | size | worst stage cosine | sigmoid max abs diff | decisions flipped at 0.5 |
|---|---|---|---|---|
| f32 | 471 MB | 1.0000000 | 7.5e-7 | 0 |
| f16 | 237 MB | 0.9999999 | 1.4e-3 | 0 |
| q8_0 | 134 MB | 0.9999154 | 1.7e-2 | 1 |
| q4_0 | 80 MB | 0.9785978 | 2.3e-1 | 7 |

One 15 s clip and a flip count are not a diarization error rate. Q8_0 looks free; Q4_0 is visibly
lossy at the last FastConformer block, so do not ship it before a DER run on real meeting data.
(These use gguf-py's reference quantizer; cera's own may round differently.)

## What the golden captures

* Offline: mel, pre-encode embeddings, the x-scaled encoder input, all 17 block outputs,
  `encoder_proj`, all 18 Transformer layer outputs, the sigmoids.
* Streaming, four presets: the checkpoint default (188/1/1, no FIFO), the model card's low latency
  (6/1/7, FIFO 188), and two tiny-cache presets (`tiny`, `tiny_nofifo`) that make a 15 s clip overflow
  the speaker cache so that compression, the silence profile and FIFO pop are exercised. Per step:
  frame bounds, chunk predictions, and the state after the update (cache, cache predictions, FIFO,
  FIFO predictions, mean silence embedding, silence count). The unrolled loop is asserted equal to
  NeMo's own `forward_streaming`, and two runs are asserted identical (so
  `torch.topk(sorted=False)` ties do not make the trace flaky).
* The generator asserts the mel front-end defaults the converter hard-codes (pre-emphasis 0.97,
  log guard 2^-24, power 2, pad_to 16, normalize NA). That the exported window and filterbank match the
  checkpoint's is shown by `verify_gguf.py`: its mel stage differs from the golden by exactly 0.

Facts a port must match, all read from NeMo 3.0.0 rather than assumed:

* `xscaling` multiplies by sqrt(512) inside `RelPositionalEncoding.forward`, after pre-encode. The
  cached speaker-cache/FIFO embeddings are pre-xscale pre-encode outputs.
* The step runs the full FastConformer over `[spkcache, fifo, chunk]` pre-encode embeddings, then
  `encoder_proj`, then the Transformer, then relu -> `first_hidden_to_hidden` -> relu ->
  `single_hidden_to_spks` -> sigmoid; chunk predictions are sliced out at `lc`/`rc`.
* Mel is computed over the whole clip and then sliced into chunks (`pad_to=16` pads the feature
  length); an incremental mel front end has to reproduce the clip-level result at chunk edges.
* The default `async_streaming=False` path is what is captured.

## The Rust model

`cera::model::sortformer` (`SortformerModel`, `SortformerStream`, `StreamingParams`) loads these
GGUFs and runs the whole pipeline on the CPU: NeMo's mel (the checkpoint's own window and
filterbank, no per-feature normalization), the stem and the 17 FastConformer blocks (shared with the
LFM2-Audio encoder through `audio_encoder::conformer_block_forward`), `encoder_proj`, the 18-layer
Transformer, the speaker head, and a port of NeMo's streaming update (FIFO, arrival-order speaker
cache with its importance scores and compression, silence profile). Use `diarize_offline` for one
pass over a clip, `diarize_streaming` for NeMo's chunked loop over a clip, or `new_live` for live
audio: `SortformerLive::push_audio` takes PCM in pieces of any size (an incremental `MelStream`
inside) and returns each frame's speaker activities once they are final. `new_stream` + `step`
is the lower-level loop that `SortformerLive` drives. `cera::speaker_labeler` then attaches
those speakers to transcribed utterances.

`cera/tests/sortformer_parity.rs` pins it to NeMo (all measured on the committed clip, f32 GGUF):

* Every stage fed NeMo's own input: mel 1.8e-4, stem 2.1e-4, all 17 FastConformer blocks <= 2.8e-5, the
  Transformer layers <= 5.7e-6, sigmoids 1.3e-6 (max abs difference; cosine 1.0000000 throughout).
* End to end from PCM: offline sigmoids 1.5e-6; Q8_0 2.7e-2 with 1 decision flipped of 772.
* Streaming, four presets (default, `tiny`, `tiny_nofifo`, and the 33-step low-latency preset, which
  is `#[ignore]`d: `--ignored`): predictions within 2.2e-6 of NeMo's, and for the two tiny presets the speaker
  cache, FIFO, predictions and silence profile match after every step (<= 3.7e-4, the mel front end's
  own difference), through 16 and 12 compressed-cache steps with 19 and 25 silence frames profiled.

Mutation checks (each makes a test fail): strong-boost scale, the latest-frame boost, the silence
disabling, the silence threshold, the `min_pos` boundary, ceil-vs-floor on the right context.

Live audio: `SortformerLive` gives bit-identical predictions to the unpadded streaming loop for
any way of cutting the audio, and its `MelStream` the whole-clip mel exactly
(`mel_stream_is_bit_identical_to_the_whole_clip_mel`, `live_matches_the_offline_streaming_loop_and_nemo`,
`live_releases_a_chunk_when_its_lookahead_arrives`, `live_buffers_stay_bounded`).

Not covered: the asynchronous NeMo update path (variable-length batches; the sync path is what a single
stream uses); non-f32 streaming (Q8_0 is checked offline only); the first-compression case
is pinned by its own hermetic test and by the tiny presets.
