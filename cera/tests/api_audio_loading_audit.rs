//! Loading and streaming audio regressions using synthetic GGUF models.
use cera::audio_pipeline::{AudioPipelineBuilder, AudioPipelineConfig};
use cera::convert::writer::{GGML_TYPE_F32, GgufWriter};
use cera::vad::VadConfig;
use cera::{BackendPreference, CeraEngine, EngineConfig, ModelLoader, ModelSource};
use std::sync::Arc;

fn finish(mut writer: GgufWriter, tensors: Vec<(String, Vec<u64>, usize)>) -> Arc<[u8]> {
    for (name, shape, len) in &tensors {
        writer.add_tensor(name.clone(), shape.clone(), GGML_TYPE_F32, len * 4);
    }
    let mut bytes = Vec::new();
    writer.write_header_and_tensor_info(&mut bytes).unwrap();
    for (_, _, len) in tensors {
        writer
            .write_tensor_data(&mut bytes, &vec![0u8; len * 4])
            .unwrap();
    }
    bytes.into()
}
fn dense(arch: &str) -> Arc<[u8]> {
    let mut w = GgufWriter::new();
    w.add_string("general.architecture", arch);
    w.add_string_array("tokenizer.ggml.tokens", vec!["a".into(), "b".into()]);
    for (key, v) in [
        ("block_count", 1),
        ("embedding_length", 32),
        ("feed_forward_length", 32),
        ("attention.head_count", 1),
        ("attention.head_count_kv", 1),
        ("context_length", 64),
        ("vocab_size", 2),
    ] {
        w.add_u32(format!("{arch}.{key}"), v);
    }
    let mut tensors = vec![
        ("token_embd.weight".into(), vec![32, 2], 64),
        ("output_norm.weight".into(), vec![32], 32),
    ];
    for norm in [
        "attn_norm",
        "ffn_norm",
        "attn_post_norm",
        "ffn_post_norm",
        "attn_q_norm",
        "attn_k_norm",
    ] {
        tensors.push((format!("blk.0.{norm}.weight"), vec![32], 32));
    }
    for proj in [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ] {
        tensors.push((format!("blk.0.{proj}.weight"), vec![32, 32], 1024));
    }
    finish(w, tensors)
}
fn vad() -> Arc<[u8]> {
    let mut w = GgufWriter::new();
    w.add_string("general.architecture", "silero_vad");
    let mut t = Vec::new();
    for (rate, stft, channels) in [("16k", 258 * 256, 129), ("8k", 130 * 128, 65)] {
        t.push((format!("stft.{rate}.basis"), vec![stft as u64], stft));
        for (i, ins, outs) in [(0, channels, 128), (1, 128, 64), (2, 64, 64), (3, 64, 128)] {
            t.push((
                format!("encoder.{rate}.{i}.weight"),
                vec![(ins * outs * 3) as u64],
                ins * outs * 3,
            ));
            t.push((format!("encoder.{rate}.{i}.bias"), vec![outs as u64], outs));
        }
        for name in ["weight_ih", "weight_hh"] {
            t.push((
                format!("decoder.{rate}.rnn.{name}"),
                vec![512 * 128],
                512 * 128,
            ));
        }
        for name in ["bias_ih", "bias_hh"] {
            t.push((format!("decoder.{rate}.rnn.{name}"), vec![512], 512));
        }
        t.push((format!("decoder.{rate}.head.weight"), vec![128], 128));
        t.push((format!("decoder.{rate}.head.bias"), vec![1], 1));
    }
    finish(w, t)
}
fn oscillating_vad() -> Arc<[u8]> {
    let bytes = vad();
    let gguf = cera::gguf::GgufFile::from_bytes(bytes.clone()).unwrap();
    let mut raw = bytes.to_vec();
    for (name, index, value) in [
        ("decoder.16k.rnn.bias_ih", 256, 1f32),
        ("decoder.16k.rnn.weight_hh", 256 * 128, -10f32),
        ("decoder.16k.head.weight", 0, 10f32),
    ] {
        let start = gguf.tensors[name].offset as usize + index * 4;
        raw[start..start + 4].copy_from_slice(&value.to_le_bytes());
    }
    raw.into()
}
fn kws() -> Arc<[u8]> {
    let mut w = GgufWriter::new();
    w.add_string("general.architecture", "kws");
    for (key, n) in [
        ("sample_rate", 16000),
        ("window_samples", 640),
        ("hop_samples", 160),
        ("mel_bins", 1),
        ("mel_window_samples", 400),
        ("mel_hop_samples", 160),
        ("fft_size", 512),
        ("embedding_dim", 1),
    ] {
        w.add_u32(format!("kws.{key}"), n);
    }
    let mut t = Vec::new();
    for (i, ins, outs) in [(0, 1, 64), (1, 64, 64), (2, 64, 64), (3, 64, 1)] {
        t.push((
            format!("kws.backbone.conv{i}.weight"),
            vec![(ins * outs * 3) as u64],
            ins * outs * 3,
        ));
        t.push((
            format!("kws.backbone.conv{i}.bias"),
            vec![outs as u64],
            outs,
        ));
    }
    for (name, n) in [
        ("dense1.weight", 32),
        ("dense1.bias", 32),
        ("dense2.weight", 32),
        ("dense2.bias", 1),
    ] {
        t.push((format!("kws.head.{name}"), vec![n as u64], n));
    }
    finish(w, t)
}

use cera::audio_pipeline::AudioPipelineEvent;
fn vad_config() -> AudioPipelineConfig {
    AudioPipelineConfig {
        auto_transcribe: false,
        pre_roll_ms: 0,
        vad_config: VadConfig {
            threshold: 0.6,
            neg_threshold: 0.55,
            min_silence_duration_ms: 0,
            speech_pad_ms: 0,
            ..Default::default()
        },
        ..Default::default()
    }
}
fn vad_pipeline(
    config: AudioPipelineConfig,
    weights: Arc<[u8]>,
) -> cera::audio_pipeline::AudioPipeline {
    AudioPipelineBuilder::new()
        .with_config(config)
        .with_vad_from_bytes(weights)
        .unwrap()
        .build()
        .unwrap()
}
#[test]
fn explicit_loader_accepts_supported_dense_families() {
    for arch in [
        "llama", "nanbeige", "minicpm", "olmo2", "mistral3", "gemma2",
    ] {
        let bytes = dense(arch);
        let cfg = EngineConfig {
            backend: BackendPreference::Cpu,
            context_size: 64,
            ..Default::default()
        };
        assert!(
            CeraEngine::from_bytes(bytes.clone(), cfg.clone()).is_ok(),
            "legacy {arch}"
        );
        assert!(
            ModelLoader::new(ModelSource::bytes(bytes))
                .config(cfg)
                .build_generative()
                .is_ok(),
            "typed {arch}"
        );
    }
}
#[test]
fn vad_segments_use_global_clock_and_exact_pcm_slices() {
    let mut p = vad_pipeline(vad_config(), oscillating_vad());
    let events = p.process_chunk(&[0.; 2048]).unwrap();
    let spans: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AudioPipelineEvent::SpeechEnd {
                start_sample,
                end_sample,
                ..
            } => Some((*start_sample, *end_sample)),
            _ => None,
        })
        .collect();
    assert_eq!(spans, [(0, 512), (1024, 1536)]);
    assert_eq!(p.current_sample(), 2048);
    assert_eq!(p.last_utterance().len(), 512);
    let events = p.process_chunk(&[0.; 1024]).unwrap();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AudioPipelineEvent::SpeechStart { sample: 2048, .. }))
    );
}
#[test]
fn vad_events_are_independent_of_input_chunk_boundaries_and_overlap_stride() {
    for stride in [512, 320] {
        let mut cfg = vad_config();
        cfg.vad_config.frame_stride = Some(stride);
        let weights = oscillating_vad();
        let pcm = vec![0.; 4096];
        let mut reference = vad_pipeline(cfg.clone(), weights.clone());
        let mut expected = reference.process_chunk(&pcm).unwrap();
        expected.extend(reference.flush().unwrap());
        for width in [1, 79, 320, 512, 2048] {
            let mut p = vad_pipeline(cfg.clone(), weights.clone());
            let mut events = Vec::new();
            for chunk in pcm.chunks(width) {
                events.extend(p.process_chunk(chunk).unwrap());
            }
            events.extend(p.flush().unwrap());
            assert_eq!(events, expected, "stride={stride}, width={width}");
            assert_eq!(p.last_utterance(), reference.last_utterance());
        }
    }
}
#[test]
fn flush_then_new_speech_keeps_global_time() {
    let mut cfg = vad_config();
    cfg.vad_config.threshold = 0.4;
    cfg.vad_config.neg_threshold = 0.1;
    let mut p = vad_pipeline(cfg, vad());
    p.process_chunk(&[0.; 512]).unwrap();
    p.flush().unwrap();
    let events = p.process_chunk(&[0.; 512]).unwrap();
    assert!(matches!(
        events.as_slice(),
        [AudioPipelineEvent::SpeechStart { sample: 512, .. }]
    ));
}
fn hotword_config() -> AudioPipelineConfig {
    AudioPipelineConfig {
        require_hotword: true,
        auto_transcribe: false,
        pre_roll_ms: 0,
        hotword_config: Some(cera::hotword::HotwordConfig {
            threshold: 0.4,
            step_ms: 10,
            ..Default::default()
        }),
        ..Default::default()
    }
}
#[test]
fn wake_detection_retains_suffix_and_applies_pipeline_config() {
    let weights = kws();
    let cfg = hotword_config();
    let pcm: Vec<_> = (0..2240).map(|i| i as f32 / 2240.).collect();
    let make = || {
        AudioPipelineBuilder::new()
            .with_config(cfg.clone())
            .with_hotword_from_bytes(weights.clone(), None)
            .unwrap()
            .build()
            .unwrap()
    };
    let mut reference = make();
    let mut expected = reference.process_chunk(&pcm).unwrap();
    expected.extend(reference.flush().unwrap());
    assert!(matches!(
        expected[0],
        AudioPipelineEvent::WakeWordDetected {
            sample_offset: 160,
            ..
        }
    ));
    assert_eq!(reference.last_utterance(), &pcm[160..]);
    for width in [37, 160, 641, 2240] {
        let mut p = make();
        let mut actual = Vec::new();
        for chunk in pcm.chunks(width) {
            actual.extend(p.process_chunk(chunk).unwrap());
        }
        actual.extend(p.flush().unwrap());
        assert_eq!(actual, expected, "width={width}");
        assert_eq!(p.last_utterance(), &pcm[160..]);
    }
}
#[test]
fn explicit_hotword_settings_override_pipeline_defaults() {
    let mut explicit = hotword_config().hotword_config.unwrap();
    explicit.threshold = 0.9;
    let mut p = AudioPipelineBuilder::new()
        .with_config(hotword_config())
        .with_hotword_from_bytes(kws(), Some(explicit))
        .unwrap()
        .build()
        .unwrap();
    assert!(p.process_chunk(&[0.1; 2240]).unwrap().is_empty());
}

#[test]
fn repeated_wake_preroll_contains_only_current_epoch_audio() {
    let mut cfg = hotword_config();
    cfg.pre_roll_ms = 20;
    cfg.hotword_config.as_mut().unwrap().cooldown_ms = 0;
    for width in [1, 77, 200] {
        let mut pipeline = AudioPipelineBuilder::new()
            .with_config(cfg.clone())
            .with_hotword_from_bytes(kws(), None)
            .unwrap()
            .build()
            .unwrap();
        pipeline.process_chunk(&[0.1; 160]).unwrap();
        pipeline.process_chunk(&[0.5; 840]).unwrap();
        pipeline.flush().unwrap();
        assert_eq!(pipeline.last_utterance().len(), 1000);
        let mut events = Vec::new();
        for chunk in [0.9; 200].chunks(width) {
            events.extend(pipeline.process_chunk(chunk).unwrap());
        }
        events.extend(pipeline.flush().unwrap());
        assert_eq!(pipeline.last_utterance(), &[0.9; 200]);
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AudioPipelineEvent::SpeechStart { sample: 1000, .. }))
        );
        assert!(events.iter().any(|event| matches!(
            event,
            AudioPipelineEvent::SpeechEnd {
                start_sample: 1000,
                end_sample: 1200,
                ..
            }
        )));
    }
}

#[test]
fn wake_cooldown_accounts_for_time_spent_in_speech() {
    let mut cfg = hotword_config();
    cfg.hotword_config.as_mut().unwrap().cooldown_ms = 100;
    let mut pipeline = AudioPipelineBuilder::new()
        .with_config(cfg)
        .with_hotword_from_bytes(kws(), None)
        .unwrap()
        .build()
        .unwrap();
    pipeline.process_chunk(&[0.1; 160]).unwrap();
    pipeline.process_chunk(&[0.5; 840]).unwrap();
    pipeline.flush().unwrap();
    // Detection at 10ms has a 100ms cooldown. Speech consumed 52.5ms;
    // resuming must neither clear the cooldown nor pause its clock.
    let mut events = pipeline.process_chunk(&[0.9; 1000]).unwrap();
    events.extend(pipeline.flush().unwrap());
    let wakes: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AudioPipelineEvent::WakeWordDetected { sample_offset, .. } => Some(*sample_offset),
            _ => None,
        })
        .collect();
    assert_eq!(wakes, [1800]); // First 10ms hop after the global 110ms deadline.
}
#[test]
fn duration_cap_applies_to_initial_vad_chunk_and_retains_final_full_segment() {
    let mut cfg = vad_config();
    cfg.max_utterance_ms = 1000;
    cfg.vad_config.threshold = 0.4;
    cfg.vad_config.neg_threshold = 0.1;
    let mut p = vad_pipeline(cfg, vad());
    let events = p.process_chunk(&[0.; 32000]).unwrap();
    let spans: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AudioPipelineEvent::SpeechEnd {
                start_sample,
                end_sample,
                ..
            } => Some((*start_sample, *end_sample)),
            _ => None,
        })
        .collect();
    assert_eq!(spans, [(0, 16000), (16000, 32000)]);
    p.flush().unwrap();
    assert_eq!(p.last_utterance().len(), 16000);
}
#[test]
fn duration_cap_without_vad_is_exact_and_chunk_independent() {
    let config = AudioPipelineConfig {
        auto_transcribe: false,
        pre_roll_ms: 0,
        max_utterance_ms: 500,
        ..Default::default()
    };
    let pcm = vec![0.1; 20123];
    let mut reference = AudioPipelineBuilder::new()
        .with_config(config.clone())
        .build()
        .unwrap();
    let mut expected = reference.process_chunk(&pcm).unwrap();
    expected.extend(reference.flush().unwrap());
    for width in [53, 8000, 20123] {
        let mut p = AudioPipelineBuilder::new()
            .with_config(config.clone())
            .build()
            .unwrap();
        let mut events = Vec::new();
        for chunk in pcm.chunks(width) {
            events.extend(p.process_chunk(chunk).unwrap());
        }
        events.extend(p.flush().unwrap());
        assert_eq!(events, expected);
        assert_eq!(p.last_utterance().len(), 4123);
    }
}
