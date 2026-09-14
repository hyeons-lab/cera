# Run the audio loading example

This example runs the **private loading prototype**, using complete synthetic
CPU GGUFs. It loads an LFM2 primary, an audio input encoder and a vocoder, releases
the model handle, ingests mono PCM, then produces text tokens and audio samples
through the retained session. The weights exercise real computations but are
not trained to transcribe or synthesize speech. Do not use this output to assess
speech quality. The public API and Leap `ModelRunner` facade remain separate gates.

The complete [audio walkthrough](../../tests/api_loading/consumer/src/bin/audio_walkthrough.rs)
implements `ModalitySink::on_audio_frames` and checks finite 24 kHz output.
Its essential loading and input sequence is:

```rust
use cera::engine::loading_prototype::{ModelLoader, ModelSource};
use cera::manifest::InferenceType;
use cera::{ModelBytes, SessionConfig};

let mut parts = ModelBytes::text(primary_bytes);
parts.multimodal_projector = Some(encoder_bytes);
parts.audio_decoder = Some(vocoder_bytes);
parts.inference_type = Some(InferenceType::LlamaCppLfm2AudioV1);
let model = ModelLoader::new(ModelSource::parts(parts))
    .config(load_config)
    .build_generative()?;
let mut session = model.create_session(SessionConfig::default())?;
drop(model);
session.append_audio(&mono_pcm, 16_000)?;
session.append_tokens(&[0, 1])?;
session.generate(&options, &mut output)?;
```

The linked program supplies the bytes, CPU configuration, samples, options and
sink. `[0, 1]` belongs to its two-token fixture. With a trained model, tokenize
and format input according to that model's audio/chat protocol. The existing
interleaved generation loop starts an audio segment after six text tokens;
`max_tokens` counts text tokens, so it is not an audio-frame limit. This example
uses a one-code decoder vocabulary to make the fixed audio sampler deterministic.

From the worktree root, export the fixture into a new temporary directory:

```bash
export PATH="/opt/homebrew/bin:$PATH"
export CARGO_TARGET_DIR=/Users/dberrios/development/cera/target
cargo test -p cera --lib export_audio_walkthrough_fixture --locked --offline \
  -- --ignored --nocapture
```

The command prints `AUDIO_EXAMPLE_DIR=...`. It intentionally retains that new
directory for the external program; ordinary test runs skip this export helper.
No model download is needed. Set `AUDIO_EXAMPLE_DIR` to that printed path.
Run the [loading probe](../../tests/api_loading/README.md) to prepare and validate
its visibility-only workspace, then set `LOADING_RUN` to its printed run directory:

```bash
export AUDIO_EXAMPLE_DIR=/absolute/path/printed/by/the/export
export LOADING_RUN=/absolute/path/to/tests/api_loading/build/run-...
export CERA_GIT_SHA=loading-probe
export CARGO_TARGET_DIR="$(python3 -c 'import json,os; print(json.load(open(os.environ["LOADING_RUN"]+"/results.json"))["target"])')"
cargo run --manifest-path "$LOADING_RUN/workspace/Cargo.toml" \
  -p loading-consumer --bin audio_walkthrough --locked --offline \
  -- "$AUDIO_EXAMPLE_DIR"
```

The program reports consumed audio positions, six text tokens, a nonzero PCM
sample count and the final session position. This is a separate run from the
probe's 21 command expectations. Swift and Kotlin's existing examples currently
exercise text loading/continuation; they do not establish foreign audio support.

## Source selection and errors

These rules describe the current loader contract, preserved by the private API:

| Component | Multipart bytes | Files, manifest or directory |
| --- | --- | --- |
| Audio input encoder | Projector is parsed as audio only in audio mode | Projector is loaded as audio only in audio mode |
| Output decoder | Dedicated vocoder; parsed projector fallback if vocoder is absent or fails GGUF parsing | Dedicated vocoder only; requires audio mode |
| Detokenizer | First successfully parsed weights: vocoder, tokenizer, parsed projector | First successfully parsed weights: vocoder, tokenizer; requires audio mode |
| Hidden-size mismatch | Drops output decoder and detokenizer | Same |

A vocoder that parses as GGUF but lacks decoder weights blocks the byte decoder's
projector fallback. Detokenizer fallback still tries subsequent sources, but the
final assembly discards it when no compatible decoder remains. A decoder without
a detokenizer can remain in the engine; sessions auto-attach output only when
both exist. Explicit text mode skips projector parsing entirely, disabling both
byte projector fallbacks. Dedicated byte vocoders can still attach under that
text declaration even though `audio_out` is false; filesystem audio loading is gated by audio mode.
Use explicit audio mode for predictable behavior across sources.

With a plain `lfm2` primary, an inferred multipart source with a parseable
projector declares image mode; it does not inspect that projector to infer audio.
Capability flags describe the declared modality and do not prove that usable
weights loaded. Missing local files, corrupt or incomplete optional components preserve
text inference. Remote asset download failures are fatal before optional parsing;
see the [remote examples](API_RESHAPE_REMOTE_EXAMPLES.md). `append_audio` reports a missing encoder or encoder/model
hidden-size mismatch before changing the session position. Automatic audio
resampling accepts supported rates; the executable tests include 8 kHz to 16 kHz.

Run the hermetic loading and ownership tests:

```bash
cargo test -p cera --lib engine::loading_prototype::tests::auxiliary::audio --locked --offline
cargo test -p cera --no-default-features --lib engine::loading_prototype::tests::auxiliary::audio --locked --offline
```

The [input tests](../../cera/src/engine/loading_prototype/tests/auxiliary/audio/input.rs)
compare real PCM encoding with independently loaded embedding controls. The
[output tests](../../cera/src/engine/loading_prototype/tests/auxiliary/audio/output.rs)
compare depthformer outputs, code embeddings, spectra and PCM, and execute loaded
sessions after parent release. Path cases remove source files before execution
on Unix and retain the directory until mappings drop on other platforms. The
[fixtures](../../cera/src/engine/loading_prototype/tests/auxiliary/audio/fixture.rs)
contain full tensor payloads, including one Conformer block, one depthformer block
and the detokenizer's eight-block layout. Device execution, speech quality,
foreign audio consumers, remote audio execution and performance remain unverified.
