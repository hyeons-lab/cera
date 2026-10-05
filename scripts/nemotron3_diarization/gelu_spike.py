"""Phase 3: GELU spike - erf (reference) vs tanh (NPU-composable) through
offline AND streaming inference on the committed clip.

Decision rule: tanh is viable for the NPU if streaming sigmoids agree closely
and zero speaker decisions flip at 0.5.
"""
import copy

import soundfile as sf
import torch
import torch.nn as nn
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

D = "/Users/dberrios/.leap/models/nemotron3-diarization"
CLIP = "cera/tests/fixtures/sortformer/clip.wav"

processor = AutoProcessor.from_pretrained(D)
erf = AutoModelForAudioFrameClassification.from_pretrained(D)
erf.eval()
tanh = copy.deepcopy(erf)
for layer in tanh.model.audio_tower.layers:
    layer.mlp.activation_fn = nn.GELU(approximate="tanh")
tanh.eval()

pcm, sr = sf.read(CLIP, dtype="float32")


def report(tag, a, b):
    pa, pb = a.sigmoid(), b.sigmoid()
    d = (pa - pb).abs()
    flips = ((pa > 0.5) != (pb > 0.5)).sum().item()
    n = pa.numel()
    print(f"{tag}: max-abs={d.max().item():.3e} mean-abs={d.mean().item():.3e} "
          f"flips={flips}/{n}")
    return flips


with torch.inference_mode():
    inputs = processor(pcm, sampling_rate=sr)
    lo_e = erf(**inputs).logits
    lo_t = tanh(**inputs).logits
    f_off = report("offline", lo_e, lo_t)

    # streaming, low_latency preset, full session incl. last chunk
    processor.set_streaming_mode("low_latency")
    n0 = processor.num_samples_first_audio_chunk
    chunks = [(pcm[:n0], True, False)]
    step_mel = processor.num_mel_frames_per_step
    mel_idx = step_mel
    start = processor.audio_chunk_start(mel_idx)
    per = processor.num_samples_per_audio_chunk
    while start + per <= len(pcm):
        chunks.append((pcm[start:start + per], False, False))
        mel_idx += step_mel
        start = processor.audio_chunk_start(mel_idx)
    chunks.append((pcm[start:], False, True))

    def run_stream(model):
        cache, outs = None, []
        for audio, first, last in chunks:
            ins = processor(audio, sampling_rate=sr, is_streaming=True,
                            is_first_audio_chunk=first, is_last_audio_chunk=last)
            out = model(**ins.to(model.device, dtype=model.dtype), speaker_cache=cache)
            outs.append(out.logits)
            cache = out.speaker_cache
        return torch.cat(outs, dim=1)

    ls_e = run_stream(erf)
    ls_t = run_stream(tanh)
    print("streamed frames:", ls_e.shape[1], "offline frames:", lo_e.shape[1])
    f_stream = report("stream ", ls_e, ls_t)
    # streaming-vs-offline self-consistency of the reference (context, not gate)
    m = min(ls_e.shape[1], lo_e.shape[1])
    report("erf stream-vs-offline", ls_e[:, :m], lo_e[:, :m])

print("VIABLE" if f_off == 0 and f_stream == 0 else "NOT VIABLE", "for tanh on NPU")
