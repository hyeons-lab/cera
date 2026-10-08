# d1 decision models

[`LiquidAI/d1-omni-600M`](https://huggingface.co/LiquidAI/d1-omni-600M) answers named questions
about a state with **zero output tokens**: every answer is read from the model's distribution over
the options. cera runs it on the text path today: a state (a string or any JSON) and one or more
questions in, typed answers out. Images and speech are not wired in yet.

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
```

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
bidirectional LFM2 embedding models) and runs wherever the engine does; the decision head runs on
the host over the trunk's per-token hidden states and scores the state at each `<mask>`. In the
head's last layer only the marker rows are computed, which gives the same scores as computing
every row.

A `<|name|>` in caller text is rewritten to `<¦name¦>` before tokenizing, so a state can never
forge a delimiter. A state that does not fit is cut on the right; the options are never cut below
their budget (see `d1/prompt.rs`).

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

The vision tower (`vision.*`) and audio tower (`audio.*`) are left out of the GGUF for now.

## Precision

Measured against LiquidAI's reference implementation (`transformers`, float32, CPU) on seven text
requests (14 questions): plain, two-and-twelve-option questions, a JSON state, options with empty descriptions,
a state that tries to forge delimiters, an empty state, and a 27,000-token state that is cut.
Every request used the same number of input tokens as the reference.

| Weights | Largest probability difference | Answers that changed |
|---|---|---|
| F32 | 5e-5 | 0 |
| F16 | 8e-4 | 0 |
| Q8_0 | 1.9e-2 | 1 of 14 questions (a near-tie on an 8-level score) |

The checkpoint is trained in float32 and the reference recommends float16 on GPUs; bfloat16
changed the top answer on 0.8% of text rows. Use F16 (or F32) for decisions and Q8_0 where size
matters more than the last few percent on near-ties.

F16 and F32 trunks run through a plain float path (one layer's weights widened at a time) because
the batched integer GEMM only exists for quantized weights; Q8_0 uses that batched path. Speed
has not been measured yet.

## Not done yet

* Images and speech: the towers are not converted and the request has no media fields.
* Metal, wgpu and NPU: the trunk is the ordinary LFM2 model, but the decision head is
  host-side and only the CPU float path is validated for F16/F32 weights.
