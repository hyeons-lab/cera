# d1 decision models

[`LiquidAI/d1-omni-600M`](https://huggingface.co/LiquidAI/d1-omni-600M) answers named questions
about a state with **zero output tokens**: every answer is read from the model's distribution over
the options. cera runs it on text, images and speech: a state (a string or any JSON, or none when
the image or the clip is the whole state) and one or more questions in, typed answers out.

## Use

```bash
# 1. Convert the SafeTensors checkpoint (F16 keeps the reference's answers; Q8_0 is smaller)
cera convert --input ./d1-omni-600M --output d1-omni-f16.gguf --quant f16

# 2. Ask
cat > request.json <<'EOF'
{
  "state": "I was charged twice this month, please refund one of them.",
  "questions": {
    "refund":  {"type": "noul",   "instructions": "Is the customer asking for a refund?"},
    "team":    {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"billing": "Charges, refunds, invoices",
                             "technical": "App or site faults",
                             "fraud": "Suspected unauthorised use"}},
    "urgency": {"type": "score",  "instructions": "How urgent is this?",
                "criteria": ["Can wait", "Today", "Blocking the customer now"]}
  }
}
EOF
cera decide --model d1-omni-f16.gguf --request request.json

# 3. Ask about an image: it is read ahead of the text, so the question can be about the picture
cera decide --model d1-omni-f16.gguf --image cats.png --request - <<'EOF'
{"questions": {"cats": {"type": "choice", "instructions": "How many cats are there?",
                        "criteria": {"one": "One", "two": "Two", "more": "Three or more"}}}}
EOF
```

`--image` can be repeated, and images can also be sent inside the request as
`"images": ["data:image/png;base64,..."]`. They are laid out in the order given.

```bash
# 4. Ask about speech: one WAV clip (up to 30 s), also sendable as "audio": "data:audio/wav;base64,..."
cera decide --model d1-omni-f16.gguf --audio note.wav --request - <<'EOF'
{"questions": {"topic": {"type": "choice", "instructions": "What is the speaker talking about?",
                         "criteria": {"food": "Food and meals", "travel": "Travel and transport"}}}}
EOF
```

A request carries images or speech, not both.

The response has the reference implementation's shape (this is the real output of the request
above on the F16 model, trimmed to the answers):

```json
{"answers": {
   "refund":  {"type": "noul", "noul": 0.9985},
   "team":    {"type": "choice", "choice": "billing", "confidence": 0.9929,
               "probabilities": {"billing": 0.9929, "technical": 0.0054, "fraud": 0.0017}},
   "urgency": {"type": "score", "score": 1.7975, "confidence": 0.8019,
               "probabilities": {"0": 0.0044, "1": 0.1937, "2": 0.8019},
               "legend": {"0": "Can wait", "1": "Today", "2": "Blocking the customer now"}}},
 "usage": {"input_tokens": 154, "output_tokens": 0}}
```

| `type` | `criteria` | answer fields |
|---|---|---|
| `noul` (yes or no) | optional `{"true": "...", "false": "..."}` | `noul`: P(yes) |
| `choice` | `{name: description}`, at least two | `choice`, `confidence`, `probabilities` |
| `score` | a list of 2 to 10 level descriptions, lowest first | `score` (expected level), `confidence`, `probabilities`, `legend` |

Text answers are calibrated with the per-type temperatures stored in the checkpoint's
`config.json`; the converter carries them in the GGUF.

## How it runs

Each question is its own sequence: `<bos> <state> state <q> instructions (<opt> <mask> option
</opt>)* <decide>`. The trunk is a **non-causal LFM2** (the same architecture as the
bidirectional LFM2 embedding models); the decision head runs on the host over the trunk's
per-token hidden states and scores the state at each `<mask>`. In the head's last layer only the
marker rows are computed, which gives the same scores as computing every row.

A `<|name|>` in caller text is rewritten to `<¦name¦>` before tokenizing, so a state can never
forge a delimiter. A state that does not fit is cut on the right; the options are never cut below
their budget (see `d1/prompt.rs`).

### Images

An image becomes a prefix of embeddings in front of the text:

1. A large image is cut into a grid of up to ten 512 px tiles plus a thumbnail; a small one is read
   whole (LFM2-VL's smart resize and tiling).
2. Every crop goes through a SigLIP2 tower that takes any patch grid (the position table is
   resized to the crop's grid), then a 2x2 pixel unshuffle and a two-layer projector. A crop of
   `h x w` patches gives `(h / 2) * (w / 2)` embeddings.
3. The prefix attends only to itself, and the convolution at its last row does not read the first
   text row, so it depends on the image alone. Text rows read everything.

The resize is part of the model: PyTorch's antialiased bilinear on `uint8`, horizontal pass first,
each pass rounded to `uint8`, with 16-bit fixed-point weights. An emulation of this algorithm
has zero differing pixels against the reference's crops on a plain and on a tiled image, and cera
implements the same arithmetic: its end-to-end answers match the reference to 7e-6 at F32.

With media, the text of a request gets the room the image leaves, and no more than
`image_text_length` (896) tokens; a `noul` without definitions of its own is worded `false: no` /
`true: yes`; and answers are the raw softmax, because the calibration temperatures were fitted on
text only.

The trunk reads a media prefix through the plain-f32 path whatever the weights' precision, and
the images of a request are encoded once for all its questions.

### Speech

A WAV clip becomes a prefix the same way:

1. The clip is 16 kHz mono, cut to 30 s and padded to 0.5 s. A WAV at another rate is resampled
   linearly and several channels are averaged; PCM 16, 24 and 32-bit and 32-bit float are read.
2. 128 log-mel features every 10 ms (NeMo's filterbank front end), then a 17-layer FastConformer
   that subsamples by 8, an MLP adapter to the trunk's width, and a residual correction
   `x + up(GELU(down(LayerNorm(x))))`. One embedding comes out per 80 ms.
3. The encoder reads exactly `samples / 160` mel frames. The STFT produces one more; feeding it
   changes the last rows and, for some lengths, the number of rows.

The encoder is the FastConformer that `crate::model::audio_encoder` already runs for LFM2-Audio,
so the converter writes the audio tower in that layout. As with images, the text of a request gets
the room the clip leaves and at most `audio_text_length` (15,360) tokens, and the answers are the
raw softmax. After speech the questions are worded as they were trained: a `choice` option is
`option_000: <description>` (its name when it has none), a `noul` is always `false: no` /
`true: yes` whatever definitions it carries, and a request with no state has the state `{}`.

## GGUF layout

The converter writes an ordinary `lfm2` GGUF (non-causal) plus the head. `d1.*` tensors and
settings are ignored by anything that only wants the trunk.

| Checkpoint tensor | GGUF tensor |
|---|---|
| `encoder.*` | the LFM2 names (`token_embd`, `blk.N.*`, `token_embd_norm`) |
| `head.type_emb.weight` | `d1.question_type.weight` |
| `head.head.layers.N.*` | `d1.blk.N.{attn_norm,attn_qkv,attn_output,ffn_norm,ffn_up,ffn_down}.{weight,bias}` |
| `head.scorer.{0,1,3}.*` | `d1.cls.norm.*`, `d1.cls.*`, `d1.cls.output.*` |

The scorer and the question-type table stay F32 at every precision. Settings:
`d1.head.block_count`, `d1.head.attention.head_count`, `d1.head.feed_forward_length`,
`d1.head.layer_norm_epsilon`, `d1.context_length`, `d1.image_text_length`,
`d1.audio_text_length`, and `d1.temperature.keys` / `d1.temperature.values`.

| `vision.tower.vision_model.*` | `d1.v.patch_embd`, `d1.v.position_embd`, `d1.v.blk.N.{ln1,attn_q,attn_k,attn_v,attn_out,ln2,ffn_up,ffn_down}`, `d1.v.post_ln` |
| `vision.projector.linear_{1,2}` | `d1.v.mm.{1,2}` |

The patch and position tables stay F32 too. Vision settings: `d1.vision.block_count`,
`d1.vision.embedding_length`, `d1.vision.feed_forward_length`, `d1.vision.attention.head_count`,
`d1.vision.patch_size`, `d1.vision.position_side`, `d1.vision.layer_norm_epsilon`,
`d1.vision.projector_hidden_length`.

| `audio.encoder.*`, `audio.adapter.*` | the LFM2-Audio encoder layout: `a.conv1d.N`, `a.pre_encode.out`, `a.blk.N.*`, `mm.a.mlp.{0,1,3}` |
| `audio.residual.{ln,down,up}` | `d1.a.res.{norm,down,up}` |

Two things are rewritten on the way: each conformer layer's batch norm is folded into a scale and
a shift (`a.blk.N.conv_norm.{weight,bias}`), and the singleton axis of the 1-D convolution kernels
is dropped. The convolutions, the folded norms and the residual block stay F32. The encoder's
settings are the `clip.audio.*` keys its loader reads, plus `d1.audio.residual_width`.

## Precision

Measured against LiquidAI's reference implementation (`transformers`, float32, CPU). The input-token
count matched the reference on every request.

Text: 7 requests (14 questions): plain, two-and-twelve-option questions, a JSON state, options with
empty descriptions, a state that tries to forge delimiters, an empty state, and a 27,000-token state
that is cut.

| Weights | Largest probability difference | Answers that changed |
|---|---|---|
| F32 | 5e-5 | 0 |
| F16 | 8e-4 | 0 |
| Q8_0 | 1.9e-2 | 1 of 14 questions (a near-tie on an 8-level score) |

Images: 4 requests (a 640x480 photo, a 2048x1536 photo that is tiled into 6 tiles and a thumbnail,
the same photo with state text, and two images), 4 to 8 questions each, PNG input.

| Weights | Largest probability difference | Answers that changed |
|---|---|---|
| F32 | 7e-6 | 0 |
| F16 | 1.7e-3 | 0 |
| Q8_0 | 4.3e-2 | 0 |

Speech: 12 clips (a 10.4 s recording with and without state text, 3 s, and cuts of 0.2, 1, 2, 2.5,
3.3, 5.1, 7.8 and 10 s, plus the recording looped to 42 s and cut to 30 s), 2 to 3 questions each.
The lengths include the ones where the encoder's output length depends on the frame rule above.

| Weights | Largest probability difference | Answers that changed |
|---|---|---|
| F32 | 5.7e-6 | 0 |
| F16 | 1.0e-3 (7 of the clips) | 0 |
| Q8_0 | 6.2e-2 (7 of the clips) | 0 |

The checkpoint is trained in float32 and the reference recommends float16 on GPUs; bfloat16
changed the top answer on 0.8% of text rows. Use F16 (or F32) for decisions and Q8_0 where size
matters more than the last few percent.

**JPEG input** decodes to slightly different pixels than Pillow's libjpeg (the Rust decoder rounds
differently), which moves F16 image answers by up to 2e-3 instead of 1.7e-3 on these requests. A
PNG, or a JPEG decoded by the caller, avoids it.

F16 and F32 text trunks run through a plain float path (one layer's weights widened at a time)
because the batched integer GEMM only exists for quantized weights; Q8_0 text uses that batched
path. On this Mac (CPU only, one run each, so indicative): a 640x480 image with one question takes
about 2 s end to end (F32 and F16), the tiled 2048x1536 image about 8 s (F32), and the pair of
those two images about 10 s (F16).

## Backends

`cera decide --device` takes `cpu`, `metal`, `gpu` (wgpu) or `auto`. The same GGUF runs on all of
them and the CPU is the reference.

| | Trunk (text, and the rows after a prefix) | Vision tower blocks | Speech encoder | Head layers |
|---|---|---|---|---|
| CPU | yes | yes | yes | yes |
| Metal | yes | yes | yes | yes |
| wgpu | yes | yes | yes | yes (head_dim 64, desktop adapters and Adreno) |
| Hexagon NPU | refused | not run | falls back to the CPU | CPU |

The trunk is bidirectional, which the GPU LFM2 graphs were not: they are causal and ignored the
model's attention mask, so a d1 model used to load on Metal or wgpu and answer a different question
(the refund probability in the text example fell from 0.998 to 0.12). Each GPU trunk now has a
non-causal attention mode (Metal's prefill attention kernel gains a flag and the media-prefix
rule; wgpu uses a register-tiled flash kernel for head_dim 64, `attention_flash_hd64.wgsl`, with the
same rule), a centred 3-tap gated convolution, and a pass that runs all rows at once: the projections and the
feed-forward in chunks of the prefill buffers, attention and the convolution as one dispatch over
every row. A loader refuses a bidirectional checkpoint where it has no such pass (the routed
`lfm2moe` arch, and the Hexagon NPU), so `--device auto` falls back instead of answering wrongly.

Against the reference, with identical token counts (Q8_0 over the whole corpus, F16 over ten
requests that cover text, tiled images and speech):

| Backend | Q8_0 largest difference | Q8_0 flipped answers | F16 largest difference | F16 flipped |
|---|---|---|---|---|
| CPU | 6.2e-2 | 1 | 1.0e-3 | 0 |
| Metal | 5.7e-2 | 1 | 1.1e-3 | 0 |
| wgpu | 5.7e-2 | 1 | 1.7e-3 | 0 |

The flipped answer is the same near-tie on an 8-level score on every backend. CPU and Metal agree
with each other to about 1e-5 on short requests.

wgpu used to drift on long prompts with Q8_0 weights (a 13.5k-token prompt answered 0.177 where the
CPU, Metal and the reference say 0.244, 0.244 and 0.246; its F16 gap was 1.4e-2) and I first put
that down to wgpu keeping f32 activations where the CPU quantizes them to Q8_0. That was wrong: the
drift went away when its attention moved to the f32 flash kernel below (0.2441 now), so it came
from the older generic prefill attention kernel and its packed-f16 K/V, though I have not isolated
which part.

Speed (Apple M-series, one run each, so indicative; Q8_0 weights):

| Request | CPU | Metal | wgpu |
|---|---|---|---|
| a 27,000-token state (two questions) | 80 s | 14 s | 24 s |
| the tiled 2048x1536 image (two questions) | 9.1 s | 2.6 s | 3.8 s |
| a 30 s clip (two questions) | 2.6 s | 1.3 s | 1.6 s |

Where the time went. A request is the media tower, then one trunk pass per question, then the head.
The head's first layer attends from every row; done one query at a time it re-read all of K and V per
query and was bound by memory traffic (5 s per question at 13.5k rows), so on the CPU it uses the
blocked flash-attention kernel the CPU trunk uses, and on Metal the head's layers run on the GPU
(the 27,000-token request falls from 21 s to 12 s, answers within 1e-3 of the host's). The vision tower was
6.6 s of a 7 s image request on the host and is about 1.9 s on Metal; a 30 s clip's request falls from 2.6 s on
the host to 1.3 s on Metal and 1.6 s on wgpu with the encoder on the GPU. On Metal a 13.5k-row trunk pass takes about 4 s.

wgpu was the slow backend for anything long (60 s for the 27,000-token request, 7.4 s for the tiled
image) because of its attention. The existing kernels were scalar: a thread per query, or a
workgroup per eight, at 0.17 TFLOPS on an M1 Max against 1.1 for Metal's matrix-unit kernel.
`attention_flash_hd64.wgsl` writes attention like the repo's GEMM instead (two register-tiled
GEMMs with a softmax between them, 4 x 4 per thread, operands in workgroup memory) and reaches about
1.5 TFLOPS, with the same code serving the ViT, the decision head and the bidirectional trunk
(grouped-query heads and the media-prefix window included). The inner loops are unrolled by hand
because wgpu bounds every loop with a 64-bit counter, which keeps the Metal compiler from
unrolling it: the same kernel written with loops ran at 0.7 TFLOPS. With it the head runs on the
wgpu GPU as well (the 27,000-token request is 29 s with the head on the host and 21 s with it on
the GPU). The causal prefill that ordinary wgpu inference uses still has the old kernel.

Elementwise wgpu ops (bias, ReLU, GELU, add) over more than 16.7M elements, which a 4k-row head
feed-forward already is, are split into several dispatches because a dispatch is at most 65535
workgroups; before that the head's answers were quietly wrong past about 3k rows.

Weights that are neither quantized with a batched GEMM (Q4_0, Q4_1, Q8_0 and the K-quants) nor
dequantized at upload run on Metal as one GEMV dispatch per row, so F16 on Metal is correct but
slow (a 27,000-token request: 230 s against 21 s for Q8_0). wgpu dequantizes them to f32 and keeps
the batched GEMM. Use Q8_0 or Q4_0 for Metal.

A single pass needs buffers of `rows x width` floats: on wgpu the largest binding must fit the
adapter's storage-binding limit (about 128 MB on common adapters, so roughly 32k rows).

### Android (Galaxy S25 Ultra, Adreno 830, Vulkan)

Checked on a device through `cera decide --device gpu` (Q8_0). The flash attention kernel is
correct there (against a direct reference to 1e-6 up to 8192 rows, grouped-query heads and the
prefix window included) and runs at about 0.22 TFLOPS, against 0.016 for the scalar kernel Android
used before at 300 tokens (the query-tiled one was 0.002 and is why Android excluded it). The
answers match the Mac: the text example's refund probability is 0.99837 on the phone's GPU and the
Mac's CPU, a 1.5k-token prompt 0.10545 against 0.10546, a 27,000-token prompt 0.24406 against 0.24405.
The phone's own CPU gives other numbers for the longer prompts (0.0917 and 0.2439 where the GPU and
the Mac say 0.1055 and 0.2509); I did not look into that.

| Request | phone CPU | phone GPU |
|---|---|---|
| 1.5k-token state | 4.0 s | 7.1 s |
| 6.8k-token state | 31 s | 28 s |
| 27,000-token state (two questions) | not run | 128 s |
| tiled 2048x1536 image | 49 s | 20 s |
| 30 s clip | 10.5 s | 6.5 s |

(The short prompt is dominated by the GPU's fixed start-up cost, building its pipelines; the
GPU pulls ahead as the work grows, except that the phone's attention is far slower than a Mac's.)

Two things the phone does that a desktop does not, and the code now respects both. A mobile GPU's
hang detector kills a dispatch, or a submission, that runs for seconds: one 13.5k-row attention
call (about 3.3 s) lost the device, and so did a 13.5k-row pass built as a single command buffer
(tens of seconds of work). A flash-attention call is therefore issued as several dispatches of at
most 2048 queries and 2^25 query-key pairs, and a trunk pass over more than 1024 rows submits
after each chunk phase and each attention dispatch. The flash kernel is on for Adreno and for
every desktop adapter (`GpuContext::supports_flash_attention`); other Android GPUs keep the older
kernels until someone checks them.

## Not done yet

* The Hexagon NPU refuses a bidirectional checkpoint (so `--device npu` reports why), and no
  bidirectional pass is written for it.
* The flash kernel is head_dim 64 only, and on Android only Adreno has been checked (an Adreno
  830); other head widths and other mobile GPUs use the older kernels.
* The media prefix (an image's or a clip's) is recomputed through the trunk for every question; it
  depends only on the media, so it could be computed once.
* `cera run --hf` and the other streaming conversions refuse a d1 checkpoint: download the
  repository and use `cera convert` on the directory.
