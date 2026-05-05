# Audio test fixtures

Real-speech WAV files used by the LFM2.5-Audio ASR integration
tests in `wick/tests/session_chain.rs`. Each file is 16 kHz mono
signed 16-bit PCM (the canonical input format `Session::append_audio`
expects) wrapped in a RIFF/WAVE container.

## Files

| File | Phrase | Anchor (case-insensitive substring) | Duration |
|---|---|---|---|
| `today_is_a_beautiful_day.wav` | "Today is a beautiful day" | `today is a beautiful day` | ~1.6 s |

The "anchor" column lists the substring the ASR test asserts on
(case-insensitive). The phrase was chosen because LFM2.5-Audio-1.5B-Q4_0
transcribes its FIRST 5 tokens word-for-word at greedy temp 0 with
system prompt `"Perform ASR."` — verified locally with `wick run`.

Q4_0 doesn't reliably emit `<|im_end|>` after the transcription,
so longer `max_tokens` runs include hallucinated continuation
(e.g. "...to be the beautiful day to be"). The substring assertion
absorbs that tail while still catching gross transcription failures
(missing words, NaN propagation, audio-encoder regressions).

## Regenerating

The committed `.wav` files were produced by macOS `say` + `afconvert`.
Run `./generate.sh` from this directory (macOS only) to reproduce.
The script is checked in alongside the binary fixtures so regenerating
on a different host or after a model swap is deterministic — `say`'s
output is stable across macOS minor versions for the same phrase + voice.

## Why external TTS instead of wick's own

The fixtures are inputs to wick's ASR path. If they were generated
by wick's own TTS, a regression in the LLM (e.g. the hidden-state
magnitude drift tracked in
`~/.claude/projects/.../memory/project_llm_magnitude_bug.md`) would
shift both the synthesis and the transcription in lockstep — the
self-loop would still match and the test would pass on garbage.
macOS `say` is a fixed external reference: a wick regression in
either direction shows up as a transcription that no longer contains
the anchor substring.
