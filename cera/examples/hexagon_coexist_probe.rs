//! Several Hexagon NPU models in one process: do they all open, how much of the DSP's address
//! space do they take together, and what do they cost each other when they run at once?
//!
//! The mix is an app that sees, thinks and listens: a vision-language model (the vision tower
//! and its LLM), a text LLM, and the streaming audio pipeline (Silero VAD, Whisper and a
//! Sortformer diarizer). The probe opens them in that order and reports the DSP bytes mapped by
//! the whole process after each open (each `HexagonContext` keeps its own driver count, so only
//! the process-wide number says how much of the shared 32-bit DSP address space is gone). Then it
//! times, with the audio fed in real time unless `--max-rate`:
//!
//! 1. each workload alone: the audio pipeline, the LLM (prefill and a long greedy decode) and the
//!    vision model (image prefill and a short decode);
//! 2. each foreground workload with the audio running, and both foreground workloads together;
//! 3. all three at once;
//! 4. the chain on the first utterances: audio, transcript, speaker, LLM reply.
//!
//! Decode stalls matter as much as rates, so every generation also reports its longest gap
//! between two tokens.
//!
//! ```text
//! cargo ndk -t arm64-v8a build --release -p cera --example hexagon_coexist_probe --features hexagon,mmap
//! adb shell 'cd /data/local/tmp/probe && ADSP_LIBRARY_PATH=$PWD/skels ./hexagon_coexist_probe \
//!     --llm llm.gguf --vlm vl.json --image photo.jpg \
//!     --vad silero_vad.gguf --whisper whisper.gguf --diarizer sortformer.gguf --wav clip.wav \
//!     [--extra-model more.gguf] [--llm-first] [--max-rate] [--only all|audio|llm] \
//!     [--prompt-tokens 700] [--reps 3] [--max-tokens 64]'
//! ```
//!
//! `--extra-model` opens one more plain text model on the NPU (and decodes a few tokens), to
//! account for a further resident model. Every line the probe wants parsed starts with `RESULT`.

#[cfg(feature = "hexagon")]
fn main() {
    probe::run();
}

#[cfg(not(feature = "hexagon"))]
fn main() {
    eprintln!("build with --features hexagon (and mmap) to run this probe");
}

#[cfg(feature = "hexagon")]
mod probe {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use cera::audio_pipeline::{AudioPipeline, AudioPipelineBuilder, AudioPipelineEvent};
    use cera::model::sortformer::SortformerModel;
    use cera::tokenizer::{ChatMessage, ChatMessageMultimodal, ContentItem, apply_chat_template};
    use cera::{
        BackendPreference, CeraEngine, EngineConfig, FinishReason, GenerateOpts, ModalitySink,
        Session, SessionConfig,
    };

    const MIB: f64 = 1048576.0;
    /// 500 ms of 16 kHz audio per call, as the always-on service feeds it.
    const CHUNK: usize = 8000;
    const CHUNK_MS: u64 = 500;
    const SILENCE_MS: usize = 1200;

    fn arg(args: &[String], key: &str) -> Option<String> {
        args.iter()
            .position(|a| a == key)
            .and_then(|i| args.get(i + 1).cloned())
    }

    fn flag(args: &[String], key: &str) -> bool {
        args.iter().any(|a| a == key)
    }

    fn mapped_mib() -> f64 {
        cera::backend::hexagon::sys::process_mapped_bytes() as f64 / MIB
    }

    fn rss_mib() -> f64 {
        std::fs::read_to_string("/proc/self/status")
            .unwrap_or_default()
            .lines()
            .find(|l| l.starts_with("VmRSS:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|kb| kb.parse::<f64>().ok())
            .map_or(f64::NAN, |kb| kb / 1024.0)
    }

    fn loadavg() -> String {
        std::fs::read_to_string("/proc/loadavg")
            .unwrap_or_default()
            .trim()
            .to_string()
    }

    fn pct(v: &[f64], p: f64) -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        let mut v = v.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f64 * p).round() as usize]
    }

    /// One line per open: what it cost and whether it worked.
    fn report_open<T>(what: &str, started: Instant, r: &Result<T, String>) {
        let status = match r {
            Ok(_) => "ok".to_string(),
            Err(e) => format!("FAILED: {e}"),
        };
        println!(
            "RESULT open {what:<12} {status} | {:.0} ms | dsp mapped {:.1} MiB | rss {:.0} MiB",
            started.elapsed().as_secs_f64() * 1e3,
            mapped_mib(),
            rss_mib()
        );
    }

    fn open_pipeline(args: &[String]) -> Result<AudioPipeline, String> {
        let mut builder = AudioPipelineBuilder::new();
        if let Some(p) = arg(args, "--vad") {
            let t = Instant::now();
            builder = builder
                .with_vad_from_file(&p)
                .map_err(|e| format!("{e:#}"))?;
            report_open("vad", t, &Ok::<_, String>(()));
        }
        if let Some(p) = arg(args, "--whisper") {
            let t = Instant::now();
            builder = builder
                .with_whisper_from_file(&p)
                .map_err(|e| format!("{e:#}"))?;
            report_open("whisper", t, &Ok::<_, String>(()));
        }
        if let Some(p) = arg(args, "--diarizer") {
            let t = Instant::now();
            let model = SortformerModel::from_file(&p).map_err(|e| format!("{e:#}"))?;
            let params = model.default_streaming().clone();
            let staged = cera::model::sortformer_hexagon::try_hexagon_sortformer(
                &model,
                params.window_frames(),
            )
            .is_some();
            builder = builder.with_diarizer(model, params);
            let name = if staged {
                "diarizer-npu"
            } else {
                "diarizer-cpu"
            };
            report_open(name, t, &Ok::<_, String>(()));
        }
        builder.build().map_err(|e| format!("{e:#}"))
    }

    fn open_engine(path: &str) -> Result<CeraEngine, String> {
        let cfg = EngineConfig {
            context_size: 4608,
            backend: BackendPreference::Hexagon,
            ..EngineConfig::default()
        };
        CeraEngine::from_path(path, cfg).map_err(|e| format!("{e:#}"))
    }

    struct Sink {
        t0: Instant,
        last: Instant,
        first_ms: Option<f64>,
        max_gap_ms: f64,
        toks: Vec<u32>,
        reason: Option<FinishReason>,
    }

    impl ModalitySink for Sink {
        fn on_text_tokens(&mut self, tokens: &[u32]) {
            let now = Instant::now();
            if self.first_ms.is_none() {
                self.first_ms = Some(now.duration_since(self.t0).as_secs_f64() * 1e3);
            } else {
                let gap = now.duration_since(self.last).as_secs_f64() * 1e3;
                self.max_gap_ms = self.max_gap_ms.max(gap);
            }
            self.last = now;
            self.toks.extend_from_slice(tokens);
        }
        fn on_done(&mut self, reason: FinishReason) {
            self.reason = Some(reason);
        }
    }

    #[derive(Default)]
    struct Gen {
        n_prompt: usize,
        prefill_ms: f64,
        ttft_ms: f64,
        n_gen: u32,
        tok_s: f64,
        max_gap_ms: f64,
        text: String,
    }

    fn new_session(engine: &CeraEngine) -> Session {
        engine
            .new_session(SessionConfig {
                max_seq_len: Some(4608),
                ..SessionConfig::default()
            })
            .expect("session")
    }

    fn render(session: &Session, system: &str, user: &str) -> Vec<u32> {
        let tok = session.tokenizer_arc();
        let msgs = vec![
            ChatMessage {
                role: "system".into(),
                content: system.into(),
            },
            ChatMessage {
                role: "user".into(),
                content: user.into(),
            },
        ];
        let mut s = apply_chat_template(&tok, &msgs, true).expect("chat template");
        // LFM2 opens a <think> block; close it empty so the model answers directly.
        if s.ends_with("<think>") {
            s.push_str("</think>\n");
        }
        tok.encode(&s)
    }

    /// A prompt of about `n` tokens: filler sentences, then an instruction that asks for a long
    /// answer, so the decode runs for the whole token budget instead of ending after a few.
    fn filler_prompt(session: &Session, n: usize) -> Vec<u32> {
        let mut body = String::new();
        let tok = session.tokenizer_arc();
        let unit = "The quick brown fox jumps over the lazy dog near the river bank. ";
        while tok.encode(&body).len() < n.saturating_sub(50) {
            body.push_str(unit);
        }
        body.push_str(
            "\nWrite a long, detailed story of at least 200 words about the animal mentioned above.",
        );
        render(session, "You are a storyteller.", &body)
    }

    /// Prefill with `prefill`, then decode greedily up to `max_tokens`.
    fn generate_with(
        session: &mut Session,
        max_tokens: u32,
        prefill: impl FnOnce(&mut Session),
    ) -> Gen {
        session.reset().expect("reset");
        session.clear_cancel();
        let tok = session.tokenizer_arc();
        let stop: Vec<u32> = tok.special_token_id("<|im_end|>").into_iter().collect();
        let opts = GenerateOpts {
            max_tokens,
            temperature: 0.0,
            top_k: 1,
            repetition_penalty: 1.0,
            flush_every_tokens: 1,
            flush_every_ms: 0,
            stop_tokens: stop,
            no_spec: true,
            ..GenerateOpts::default()
        };
        let t0 = Instant::now();
        prefill(session);
        let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
        let n_prompt = session.position();
        let mut sink = Sink {
            t0,
            last: t0,
            first_ms: None,
            max_gap_ms: 0.0,
            toks: Vec::new(),
            reason: None,
        };
        let summary = session.generate(&opts, &mut sink).expect("generate");
        Gen {
            n_prompt: n_prompt as usize,
            prefill_ms,
            ttft_ms: sink.first_ms.unwrap_or(f64::NAN),
            n_gen: summary.tokens_generated,
            tok_s: summary.decode_tok_per_sec(),
            max_gap_ms: sink.max_gap_ms,
            text: tok.decode(&sink.toks),
        }
    }

    fn generate(session: &mut Session, tokens: &[u32], max_tokens: u32) -> Gen {
        generate_with(session, max_tokens, |s| {
            s.append_tokens(tokens).expect("prefill")
        })
    }

    fn generate_vl(session: &mut Session, image: &[u8], prompt: &str, max_tokens: u32) -> Gen {
        generate_with(session, max_tokens, |s| {
            let msgs = vec![ChatMessageMultimodal {
                role: "user".into(),
                content: vec![
                    ContentItem::Image,
                    ContentItem::Text {
                        text: prompt.into(),
                    },
                ],
            }];
            s.append_chat_with_images(&msgs, &[image], true)
                .expect("image prefill")
        })
    }

    /// `clip` repeated `copies` times with silence between and after.
    fn stream_of(clip: &[f32], copies: usize) -> Vec<f32> {
        let gap = vec![0.0f32; 16 * SILENCE_MS];
        let mut s = Vec::new();
        for _ in 0..copies {
            s.extend_from_slice(clip);
            s.extend_from_slice(&gap);
        }
        s
    }

    #[derive(Default)]
    struct AudioRun {
        /// Wall time of every `process_chunk` call, ms.
        chunk_ms: Vec<f64>,
        /// The call that produced each utterance's transcript, ms (VAD end, then Whisper).
        transcribe_ms: Vec<f64>,
        transcripts: Vec<String>,
        labels: Vec<Option<u32>>,
        wall_ms: f64,
    }

    impl AudioRun {
        fn absorb(&mut self, other: AudioRun) {
            self.chunk_ms.extend(other.chunk_ms);
            self.transcribe_ms.extend(other.transcribe_ms);
            self.transcripts.extend(other.transcripts);
            self.labels.extend(other.labels);
        }
    }

    /// Feed `stream` in 500 ms chunks. Paced to real time when `realtime`: a chunk is not handed
    /// over before its audio would have been captured, but is never held back once behind.
    fn feed(
        pipeline: &mut AudioPipeline,
        stream: &[f32],
        realtime: bool,
        keep_going: &dyn Fn() -> bool,
    ) -> AudioRun {
        let mut run = AudioRun::default();
        let t_all = Instant::now();
        for (i, chunk) in stream.chunks(CHUNK).enumerate() {
            if !keep_going() {
                break;
            }
            if realtime {
                let due = t_all + Duration::from_millis(CHUNK_MS * i as u64);
                if let Some(wait) = due.checked_duration_since(Instant::now()) {
                    std::thread::sleep(wait);
                }
            }
            let t = Instant::now();
            let events = pipeline.process_chunk(chunk).expect("process_chunk");
            let ms = t.elapsed().as_secs_f64() * 1e3;
            run.chunk_ms.push(ms);
            collect(&mut run, events, ms);
        }
        let t = Instant::now();
        let events = pipeline.flush().expect("flush");
        collect(&mut run, events, t.elapsed().as_secs_f64() * 1e3);
        run.wall_ms = t_all.elapsed().as_secs_f64() * 1e3;
        run
    }

    fn collect(run: &mut AudioRun, events: Vec<AudioPipelineEvent>, call_ms: f64) {
        for e in events {
            match e {
                AudioPipelineEvent::UtteranceTranscribed { text, .. } => {
                    run.transcribe_ms.push(call_ms);
                    run.transcripts.push(text);
                }
                AudioPipelineEvent::UtteranceLabeled { speaker, .. } => run.labels.push(speaker),
                _ => {}
            }
        }
    }

    fn audio_line(tag: &str, run: &AudioRun) {
        println!(
            "RESULT audio {tag}: {} chunks | chunk ms p50 {:.1} p95 {:.1} max {:.1} | {} utterances, transcribe-call ms p50 {:.0} max {:.0} | {} labeled | load {}",
            run.chunk_ms.len(),
            pct(&run.chunk_ms, 0.5),
            pct(&run.chunk_ms, 0.95),
            pct(&run.chunk_ms, 1.0),
            run.transcripts.len(),
            pct(&run.transcribe_ms, 0.5),
            pct(&run.transcribe_ms, 1.0),
            run.labels.len(),
            loadavg()
        );
    }

    fn gen_line(kind: &str, tag: &str, gens: &[Gen]) {
        let col = |f: fn(&Gen) -> f64| gens.iter().map(f).collect::<Vec<_>>();
        println!(
            "RESULT {kind} {tag}: {} runs, {} prompt tokens, {} generated | prefill ms p50 {:.0} max {:.0} | ttft ms p50 {:.0} | decode tok/s p50 {:.1} min {:.1} | longest token gap ms p50 {:.0} max {:.0} | load {}",
            gens.len(),
            gens.first().map_or(0, |g| g.n_prompt),
            gens.first().map_or(0, |g| g.n_gen),
            pct(&col(|g| g.prefill_ms), 0.5),
            pct(&col(|g| g.prefill_ms), 1.0),
            pct(&col(|g| g.ttft_ms), 0.5),
            pct(&col(|g| g.tok_s), 0.5),
            pct(&col(|g| g.tok_s), 0.0),
            pct(&col(|g| g.max_gap_ms), 0.5),
            pct(&col(|g| g.max_gap_ms), 1.0),
            loadavg()
        );
    }

    /// Run `llm`, `vlm` and `audio` workers (each when present) at once until the foreground
    /// ones finish their `reps`; the audio loops until then. Returns each worker's results.
    #[allow(clippy::too_many_arguments)]
    fn contend(
        pipeline: Option<&mut AudioPipeline>,
        llm: Option<(&mut Session, &[u32])>,
        vlm: Option<(&mut Session, &[u8], &str)>,
        stream: &[f32],
        realtime: bool,
        reps: usize,
        max_tokens: u32,
        vl_max_tokens: u32,
    ) -> (Vec<Gen>, Vec<Gen>, AudioRun) {
        let done = AtomicBool::new(false);
        let foreground_left =
            AtomicUsize::new(usize::from(llm.is_some()) + usize::from(vlm.is_some()));
        let (mut llm_gens, mut vl_gens, mut audio_run) =
            (Vec::new(), Vec::new(), AudioRun::default());
        let finish = || {
            if foreground_left.fetch_sub(1, Ordering::SeqCst) == 1 {
                done.store(true, Ordering::Relaxed);
            }
        };
        std::thread::scope(|scope| {
            let audio = pipeline.map(|p| {
                scope.spawn(|| {
                    let mut all = AudioRun::default();
                    let t = Instant::now();
                    while !done.load(Ordering::Relaxed) {
                        all.absorb(feed(p, stream, realtime, &|| !done.load(Ordering::Relaxed)));
                    }
                    all.wall_ms = t.elapsed().as_secs_f64() * 1e3;
                    all
                })
            });
            let llm_worker = llm.map(|(s, tokens)| {
                scope.spawn(|| {
                    let g: Vec<Gen> = (0..reps).map(|_| generate(s, tokens, max_tokens)).collect();
                    finish();
                    g
                })
            });
            let vl_worker = vlm.map(|(s, image, prompt)| {
                scope.spawn(|| {
                    let g: Vec<Gen> = (0..reps)
                        .map(|_| generate_vl(s, image, prompt, vl_max_tokens))
                        .collect();
                    finish();
                    g
                })
            });
            if let Some(w) = llm_worker {
                llm_gens = w.join().expect("llm thread");
            }
            if let Some(w) = vl_worker {
                vl_gens = w.join().expect("vlm thread");
            }
            if let Some(a) = audio {
                audio_run = a.join().expect("audio thread");
            }
        });
        (llm_gens, vl_gens, audio_run)
    }

    pub fn run() {
        let args: Vec<String> = std::env::args().collect();
        let reps: usize = arg(&args, "--reps").map_or(3, |v| v.parse().unwrap());
        let prompt_tokens: usize =
            arg(&args, "--prompt-tokens").map_or(700, |v| v.parse().unwrap());
        let max_tokens: u32 = arg(&args, "--max-tokens").map_or(64, |v| v.parse().unwrap());
        let vl_max_tokens: u32 = arg(&args, "--vl-max-tokens").map_or(32, |v| v.parse().unwrap());
        let copies: usize = arg(&args, "--copies").map_or(4, |v| v.parse().unwrap());
        let realtime = !flag(&args, "--max-rate");
        let only = arg(&args, "--only").unwrap_or_else(|| "all".into());
        let want_audio = only == "all" || only == "audio";
        let want_llm = only == "all" || only == "llm";

        println!(
            "RESULT start: audio fed {} | load {} | dsp mapped {:.1} MiB | rss {:.0} MiB",
            if realtime {
                "in real time"
            } else {
                "at max rate"
            },
            loadavg(),
            mapped_mib(),
            rss_mib()
        );

        let mut pipeline: Option<AudioPipeline> = None;
        let (mut engine, mut vl_engine, mut extra): (
            Option<CeraEngine>,
            Option<CeraEngine>,
            Option<CeraEngine>,
        ) = (None, None, None);
        let open_audio = |pipeline: &mut Option<AudioPipeline>| {
            if want_audio && (arg(&args, "--vad").is_some() || arg(&args, "--whisper").is_some()) {
                let t = Instant::now();
                let r = open_pipeline(&args);
                report_open("pipeline", t, &r);
                *pipeline = r.ok();
            }
        };
        let open_models = |engine: &mut Option<CeraEngine>,
                           vl_engine: &mut Option<CeraEngine>,
                           extra: &mut Option<CeraEngine>| {
            for (key, name, slot) in [
                ("--vlm", "vlm", &mut *vl_engine),
                ("--llm", "llm", &mut *engine),
                ("--extra-model", "extra-model", &mut *extra),
            ] {
                if want_llm && let Some(p) = arg(&args, key) {
                    let t = Instant::now();
                    let r = open_engine(&p);
                    report_open(name, t, &r);
                    *slot = r.ok();
                }
            }
        };
        if flag(&args, "--llm-first") {
            open_models(&mut engine, &mut vl_engine, &mut extra);
            open_audio(&mut pipeline);
        } else {
            open_audio(&mut pipeline);
            open_models(&mut engine, &mut vl_engine, &mut extra);
        }

        let clip = arg(&args, "--wav")
            .map(|p| cera::wav::read_wav_mono_16k(&p).expect("wav"))
            .unwrap_or_default();
        let stream = stream_of(&clip, copies);
        let image = arg(&args, "--image").map(|p| std::fs::read(&p).expect("image"));
        let vl_prompt = "Describe this photo in two sentences.";

        if let Some(extra) = &extra {
            let mut s = new_session(extra);
            let toks = filler_prompt(&s, 64);
            let g = generate(&mut s, &toks, 8);
            println!(
                "RESULT extra-model decode: {} tokens at {:.1} tok/s",
                g.n_gen, g.tok_s
            );
        }

        let mut session = engine.as_ref().map(new_session);
        let prompt = session.as_ref().map(|s| filler_prompt(s, prompt_tokens));
        let mut vl_session = vl_engine.as_ref().map(new_session);
        let have_vl = vl_session.is_some() && image.is_some();

        // 1. Each workload alone.
        if let Some(p) = pipeline.as_mut() {
            let _ = feed(p, &stream, false, &|| true); // warm-up, so first-use costs stay out
            let run = feed(p, &stream, realtime, &|| true);
            audio_line("alone", &run);
            for (i, t) in run.transcripts.iter().enumerate() {
                println!(
                    "RESULT transcript[{i}]: {t:?} speaker {:?}",
                    run.labels.get(i)
                );
            }
        }
        if let (Some(s), Some(tokens)) = (session.as_mut(), prompt.as_ref()) {
            let _ = generate(s, tokens, 8);
            let alone: Vec<Gen> = (0..reps).map(|_| generate(s, tokens, max_tokens)).collect();
            gen_line("llm", "alone", &alone);
        }
        if have_vl {
            let (s, img) = (vl_session.as_mut().unwrap(), image.as_ref().unwrap());
            let _ = generate_vl(s, img, vl_prompt, 4);
            let alone: Vec<Gen> = (0..reps)
                .map(|_| generate_vl(s, img, vl_prompt, vl_max_tokens))
                .collect();
            gen_line("vlm", "alone", &alone);
            println!("RESULT vlm answer: {:?}", alone[0].text.trim());
        }

        // 2. Pairs and 3. all at once.
        let img_slice = image.as_deref();
        let combos: [(&str, bool, bool, bool); 5] = [
            ("llm + audio", true, false, true),
            ("vlm + audio", false, true, true),
            ("llm + vlm", true, true, false),
            ("llm + vlm + audio", true, true, true),
            ("audio only (control)", false, false, true),
        ];
        for (name, use_llm, use_vl, use_audio) in combos {
            let llm = if use_llm {
                session.as_mut().zip(prompt.as_deref())
            } else {
                None
            };
            let vlm = if use_vl && have_vl {
                vl_session
                    .as_mut()
                    .zip(img_slice)
                    .map(|(s, i)| (s, i, vl_prompt))
            } else {
                None
            };
            let audio = if use_audio { pipeline.as_mut() } else { None };
            if (use_llm && llm.is_none())
                || (use_vl && vlm.is_none())
                || (use_audio && audio.is_none())
            {
                continue;
            }
            if !use_llm && !use_vl {
                continue; // the control is the alone line above
            }
            let (lg, vg, ar) = contend(
                audio,
                llm,
                vlm,
                &stream,
                realtime,
                reps,
                max_tokens,
                vl_max_tokens,
            );
            if !lg.is_empty() {
                gen_line("llm", &format!("[{name}]"), &lg);
            }
            if !vg.is_empty() {
                gen_line("vlm", &format!("[{name}]"), &vg);
            }
            if use_audio {
                audio_line(&format!("[{name}]"), &ar);
            }
        }

        // 4. The chain: audio -> transcript -> speaker -> LLM, for the first utterances.
        if let (Some(p), Some(s)) = (pipeline.as_mut(), session.as_mut()) {
            let r = feed(p, &stream, false, &|| true);
            for (i, text) in r.transcripts.iter().take(2).enumerate() {
                let who = r
                    .labels
                    .get(i)
                    .copied()
                    .flatten()
                    .map_or("unknown".to_string(), |k| format!("speaker {k}"));
                let user = format!(
                    "A voice assistant heard {who} say: \"{}\"\nReply with one short sentence.",
                    text.trim()
                );
                let tokens = render(s, "You are a concise assistant.", &user);
                let g = generate(s, &tokens, 32);
                println!(
                    "RESULT chain[{i}]: transcribe call {:.0} ms | llm prompt {} tok, prefill {:.0} ms, ttft {:.0} ms, {} tokens at {:.1} tok/s | reply {:?}",
                    r.transcribe_ms.get(i).copied().unwrap_or(f64::NAN),
                    g.n_prompt,
                    g.prefill_ms,
                    g.ttft_ms,
                    g.n_gen,
                    g.tok_s,
                    g.text.trim()
                );
            }
        }

        println!(
            "RESULT end: dsp mapped {:.1} MiB | rss {:.0} MiB | load {}",
            mapped_mib(),
            rss_mib(),
            loadavg()
        );
    }
}
