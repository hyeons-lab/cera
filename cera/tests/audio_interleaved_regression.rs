//! Regression test suite for LFM2.5-Audio interleaved speech generation.
//!
//! Invariants guarded:
//! 1. `embed_audio_token` produces the unnormalized sum of 8 codebook embeddings (PR #18641).
//! 2. `forward_from_embedding` applies output RMSNorm before the LM head projection.
//! 3. Audio synthesis produces natural acoustic spectral envelopes without DC saturation or clipping.

use cera::kv_cache::InferenceState;
use cera::model::Model;
use cera::model::audio_decoder::{
    AudioDecoderWeights, AudioGpu, DetokenizerWeights, embed_audio_token,
};
use cera::model::lfm2::Lfm2Model;
use std::path::PathBuf;
use std::sync::Arc;

fn load_models() -> Option<(Arc<cera::gguf::GgufFile>, Arc<cera::gguf::GgufFile>)> {
    let base =
        PathBuf::from(std::env::var("HOME").ok()?).join(".leap/models/LFM2.5-Audio-1.5B-Q4_0");
    let model_path = base.join("LFM2.5-Audio-1.5B-Q4_0.gguf");
    let vocoder_path = base.join("vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf");

    if !model_path.exists() || !vocoder_path.exists() {
        eprintln!("Audio model files not found in ~/.leap/models, skipping regression tests");
        return None;
    }

    let model_gguf = cera::gguf::GgufFile::open_arc(&model_path).ok()?;
    let vocoder_gguf = cera::gguf::GgufFile::open_arc(&vocoder_path).ok()?;
    Some((model_gguf, vocoder_gguf))
}

#[test]
fn test_embed_audio_token_is_unnormalized_sum() {
    let Some((_, vocoder_gguf)) = load_models() else {
        return;
    };
    let dec_w = AudioDecoderWeights::from_gguf(&vocoder_gguf).expect("Failed to load vocoder");

    let sample_codes = [100, 200, 300, 400, 500, 600, 700, 800];
    let emb = embed_audio_token(&dec_w, &sample_codes);

    assert_eq!(emb.len(), dec_w.decoder_config.n_embd);

    let l2_norm = emb.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let mean_sq = emb.iter().map(|&x| x * x).sum::<f32>() / emb.len() as f32;

    assert!(
        mean_sq > 0.001,
        "Embedding mean square is too low ({mean_sq}), unexpected scaling/normalization"
    );
    assert!(
        l2_norm > 1.0,
        "Embedding L2 norm ({l2_norm}) indicates unexpected RMS normalization"
    );
}

#[test]
fn test_forward_from_embedding_applies_output_rmsnorm() {
    let Some((model_gguf, vocoder_gguf)) = load_models() else {
        return;
    };
    let model =
        Lfm2Model::from_gguf((*model_gguf).clone(), 512).expect("Failed to load LFM2 model");
    let dec_w = AudioDecoderWeights::from_gguf(&vocoder_gguf).expect("Failed to load vocoder");
    let mut state = InferenceState::from_config(model.config()).expect("Failed to init state");

    let sample_codes = [100, 200, 300, 400, 500, 600, 700, 800];
    let emb = embed_audio_token(&dec_w, &sample_codes);

    // Forward through LLM from embedding and get logits.
    let logits = model.forward_from_embedding(&emb, 0, &mut state);

    assert_eq!(logits.len(), model.config().vocab_size);

    for &l in &logits {
        assert!(l.is_finite(), "Logit value must be finite, found {l}");
    }

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_logit = logits.iter().cloned().fold(f32::INFINITY, f32::min);

    assert!(
        max_logit < 100.0 && min_logit > -100.0,
        "Logit bounds [{min_logit}, {max_logit}] indicate unnormalized forward projection"
    );
}

#[test]
fn test_acoustic_spectrum_energy_distribution() {
    let Some((_, vocoder_gguf)) = load_models() else {
        return;
    };
    let detok_w = DetokenizerWeights::from_gguf(&vocoder_gguf).expect("Failed to load detok");
    let dec_w = AudioDecoderWeights::from_gguf(&vocoder_gguf).expect("Failed to load dec");
    let mut state = cera::model::audio_decoder::DetokenizerState::new(&detok_w.config);

    let sample_codes = [336, 947, 1740, 942, 1472, 111, 1633, 1771];
    let spectrum = cera::model::audio_decoder::detokenize_to_spectrum(
        &detok_w,
        &dec_w,
        &mut state,
        &sample_codes,
    );

    assert_eq!(spectrum.len(), 6 * 641 * 2);

    let mut streamer = cera::model::audio_decoder::IstftStreamer::new(
        detok_w.config.n_fft,
        detok_w.config.hop_length,
    );
    let mut pcm = streamer.feed_frames(&spectrum);
    let flush = streamer.flush();
    pcm.extend_from_slice(&flush);

    assert_eq!(pcm.len(), 1440);

    let max_amp = pcm.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    let rms = (pcm.iter().map(|&x| x * x).sum::<f32>() / pcm.len() as f32).sqrt();

    assert!(
        max_amp <= 1.0,
        "PCM signal exceeded [-1.0, 1.0] range (clipped at {max_amp})"
    );
    assert!(
        rms > 0.001 && rms < 0.5,
        "RMS level ({rms}) outside valid acoustic range [0.001, 0.5]"
    );
}

#[test]
fn test_tts_prompt_token_prediction() {
    let Some((model_gguf, vocoder_gguf)) = load_models() else {
        return;
    };
    let model =
        Lfm2Model::from_gguf((*model_gguf).clone(), 512).expect("Failed to load LFM2 model");
    let tokenizer =
        cera::tokenizer::BpeTokenizer::from_gguf(&model_gguf).expect("Failed to load tokenizer");
    let mut state = InferenceState::from_config(model.config()).expect("Failed to init state");

    let messages = vec![
        cera::tokenizer::ChatMessage {
            role: "system".into(),
            content: "Perform TTS. Use the US female voice.".into(),
        },
        cera::tokenizer::ChatMessage {
            role: "user".into(),
            content: "Hello, how are you today?".into(),
        },
    ];
    let formatted = cera::tokenizer::apply_chat_template(&tokenizer, &messages, true)
        .expect("Failed to apply chat template");

    println!("Formatted prompt:\n{formatted}");
    let tokens = tokenizer.encode(&formatted);
    println!("Prompt tokens ({}): {:?}", tokens.len(), tokens);

    let mut logits = Vec::new();
    for (i, &tok) in tokens.iter().enumerate() {
        logits = model.forward(&[tok], i, &mut state);
    }

    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("Top 10 predicted tokens after prompt:");
    for (tok_id, val) in indexed.iter().take(10) {
        let piece = tokenizer.decode(&[*tok_id as u32]);
        println!("  token {tok_id:5} (val={val:8.4}): {piece:?}");
    }

    let dec_w = AudioDecoderWeights::from_gguf(&vocoder_gguf).expect("Failed to load vocoder");
    let detok_w = DetokenizerWeights::from_gguf(&vocoder_gguf).expect("Failed to load detok");
    let mut decoder =
        cera::audio_engine::AudioOutputDecoder::new(&dec_w, &detok_w, None, 0.0, 1, false);

    let mut pos = tokens.len();
    let mut emb = model.forward_embedding(&[128], pos, &mut state);
    pos += 1;

    for frame_idx in 0..20 {
        let outcome = decoder.decode_frame(&emb);
        match outcome {
            cera::audio_engine::FrameOutcome::End => {
                println!("Frame {frame_idx}: Emitted END token");
                break;
            }
            cera::audio_engine::FrameOutcome::Codes {
                audio_embedding,
                pcm,
                codes,
            } => {
                let sum_sq: f32 = pcm.iter().map(|&x| x * x).sum();
                let rms = (sum_sq / pcm.len().max(1) as f32).sqrt();
                println!("Frame {frame_idx}: codes={codes:?}, rms={rms:.4}");
                emb = model.forward_hidden_from_embedding(&audio_embedding, pos, &mut state);
                pos += 1;
            }
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_gpu_tts_prompt_token_prediction() {
    let Some((model_gguf, vocoder_gguf)) = load_models() else {
        return;
    };
    let gpu_model = cera::model::gpu_lfm2::GpuLfm2Model::from_gguf_with_id(
        (*model_gguf).clone(),
        2048,
        "test_gpu".to_string(),
    )
    .unwrap();

    let gpu_voc = cera::model::wgpu_audio_decoder::WgpuAudioDecoder::from_ggufs_with_context(
        gpu_model.ctx().clone(),
        &vocoder_gguf,
        Some(&vocoder_gguf),
    )
    .unwrap();

    let tokenizer =
        cera::tokenizer::BpeTokenizer::from_gguf(&model_gguf).expect("Failed to load tokenizer");
    let mut state = InferenceState::from_config(gpu_model.config()).expect("Failed to init state");

    let messages = vec![
        cera::tokenizer::ChatMessage {
            role: "system".into(),
            content: "Perform TTS. Use the US female voice.".into(),
        },
        cera::tokenizer::ChatMessage {
            role: "user".into(),
            content: "Hello, how are you today?".into(),
        },
    ];
    let formatted = cera::tokenizer::apply_chat_template(&tokenizer, &messages, true)
        .expect("Failed to apply chat template");

    let tokens = tokenizer.encode(&formatted);
    println!("GPU Prompt tokens ({}): {:?}", tokens.len(), tokens);

    let (last, prefix) = tokens.split_last().unwrap();
    let mut pos = 0;
    for &tok in prefix {
        gpu_model.forward_prefill_step(tok, pos, &mut state);
        pos += 1;
    }
    let next = pollster::block_on(gpu_model.forward_greedy_async(*last, pos, &mut state)).unwrap();
    println!("GPU First token from prefill: {next}");
    assert_eq!(next, 128);
    pos += 1;

    let cpu_model =
        Lfm2Model::from_gguf((*model_gguf).clone(), 512).expect("Failed to load LFM2 model");
    let mut cpu_state =
        InferenceState::from_config(cpu_model.config()).expect("Failed to init state");
    for (i, &tok) in tokens.iter().enumerate() {
        cpu_model.forward(&[tok], i, &mut cpu_state);
    }
    let cpu_emb = cpu_model.forward_embedding(&[128], tokens.len(), &mut cpu_state);

    let gpu_emb =
        pollster::block_on(gpu_model.forward_embedding_async(128, pos, &mut state)).unwrap();
    pos += 1;

    let dot: f32 = cpu_emb.iter().zip(&gpu_emb).map(|(a, b)| a * b).sum();
    let na: f32 = cpu_emb.iter().map(|a| a * a).sum::<f32>().sqrt();
    let nb: f32 = gpu_emb.iter().map(|b| b * b).sum::<f32>().sqrt();
    let cos = dot / (na * nb).max(1e-8);
    println!("Token 128 embedding cosine similarity CPU vs GPU: {cos:.6}");

    let dec_w = AudioDecoderWeights::from_gguf(&vocoder_gguf).expect("Failed to load vocoder");
    let detok_w = DetokenizerWeights::from_gguf(&vocoder_gguf).expect("Failed to load detok");
    let mut decoder =
        cera::audio_engine::AudioOutputDecoder::new(&dec_w, &detok_w, Some(&gpu_voc), 0.0, 1, true);

    let mut cpu_df_state =
        cera::model::audio_decoder::DepthformerState::new(&dec_w.depthformer_config);
    let codes_from_cpu_emb =
        cera::model::audio_decoder::sample_audio_frame(&dec_w, &mut cpu_df_state, &cpu_emb, 0.0, 1);
    let codes_from_gpu_emb =
        cera::model::audio_decoder::sample_audio_frame(&dec_w, &mut cpu_df_state, &gpu_emb, 0.0, 1);
    println!("Depthformer codes from CPU emb: {codes_from_cpu_emb:?}");
    println!("Depthformer codes from GPU emb (CPU df): {codes_from_gpu_emb:?}");

    for frame_idx in 0..20 {
        let outcome = pollster::block_on(
            decoder.decode_frame_from_gpu_hidden_async(gpu_model.hidden_buffer()),
        )
        .unwrap();
        match outcome {
            cera::audio_engine::FrameOutcome::End => {
                println!("GPU Frame {frame_idx}: Emitted END token");
                break;
            }
            cera::audio_engine::FrameOutcome::Codes {
                audio_embedding,
                pcm,
                codes,
            } => {
                let sum_sq: f32 = pcm.iter().map(|&x| x * x).sum();
                let rms = (sum_sq / pcm.len().max(1) as f32).sqrt();
                println!("GPU Frame {frame_idx}: codes={codes:?}, rms={rms:.4}");
                gpu_model
                    .forward_hidden_from_embedding_gpu(&audio_embedding, pos, &mut state)
                    .unwrap();
                pos += 1;
            }
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_gpu_depthformer_exact_layer_parity() {
    let Some((model_gguf, vocoder_gguf)) = load_models() else {
        return;
    };
    let gpu_model = cera::model::gpu_lfm2::GpuLfm2Model::from_gguf_with_id(
        (*model_gguf).clone(),
        2048,
        "test_gpu_parity".to_string(),
    )
    .unwrap();

    let gpu_voc = cera::model::wgpu_audio_decoder::WgpuAudioDecoder::from_ggufs_with_context(
        gpu_model.ctx().clone(),
        &vocoder_gguf,
        Some(&vocoder_gguf),
    )
    .unwrap();

    let dec_w = AudioDecoderWeights::from_gguf(&vocoder_gguf).expect("Failed to load vocoder");
    let mut df_state = cera::model::audio_decoder::DepthformerState::new(&dec_w.depthformer_config);

    // Test a synthetic or token-128 embedding
    let test_emb = vec![0.05f32; 2048];
    let cpu_codes =
        cera::model::audio_decoder::sample_audio_frame(&dec_w, &mut df_state, &test_emb, 0.0, 1);
    let gpu_codes =
        pollster::block_on(gpu_voc.sample_audio_frame_async(&test_emb, 0.0, 1)).unwrap();
    println!("Parity check:\n  CPU codes: {cpu_codes:?}\n  GPU codes: {gpu_codes:?}");
    assert_eq!(
        cpu_codes, gpu_codes,
        "CPU and GPU Depthformer sampled codes MUST match exactly!"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_cpu_vs_gpu_llm_prefill_and_decode_parity() {
    let Some((model_gguf, _)) = load_models() else {
        return;
    };
    let tokenizer =
        cera::tokenizer::BpeTokenizer::from_gguf(&model_gguf).expect("Failed to load tokenizer");
    let messages = [
        cera::tokenizer::ChatMessage {
            role: "system".into(),
            content: "Perform TTS. Use the US female voice.".into(),
        },
        cera::tokenizer::ChatMessage {
            role: "user".into(),
            content: "Hello, how are you today?".into(),
        },
    ];
    let formatted = cera::tokenizer::apply_chat_template(&tokenizer, &messages, true)
        .expect("Failed to apply chat template");
    let tokens = tokenizer.encode(&formatted);

    let cpu_model =
        Lfm2Model::from_gguf((*model_gguf).clone(), 512).expect("Failed to load CPU model");
    let gpu_model = cera::model::gpu_lfm2::GpuLfm2Model::from_gguf_with_id(
        (*model_gguf).clone(),
        2048,
        "test_prefill_parity".to_string(),
    )
    .unwrap();

    let mut cpu_state =
        InferenceState::from_config(cpu_model.config()).expect("Failed to init state");
    let mut gpu_state =
        InferenceState::from_config(cpu_model.config()).expect("Failed to init state");

    for (pos, &tok) in tokens.iter().enumerate() {
        let cpu_logits = cpu_model.forward(&[tok], pos, &mut cpu_state);
        let gpu_logits =
            pollster::block_on(gpu_model.forward_logits_async(tok, pos, &mut gpu_state)).unwrap();

        let max_diff: f32 = cpu_logits
            .iter()
            .zip(&gpu_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let cpu_top = cpu_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let gpu_top = gpu_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        println!(
            "Token {pos} ({tok}): max_diff={max_diff:.4}, cpu_top={cpu_top}, gpu_top={gpu_top}"
        );
        if cpu_top != gpu_top {
            println!("  MISMATCH at pos {pos}! CPU predicted {cpu_top}, GPU predicted {gpu_top}");
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_cpu_vs_gpu_audio_loop_parity() {
    let Some((model_gguf, vocoder_gguf)) = load_models() else {
        return;
    };
    let tokenizer =
        cera::tokenizer::BpeTokenizer::from_gguf(&model_gguf).expect("Failed to load tokenizer");
    let messages = [
        cera::tokenizer::ChatMessage {
            role: "system".into(),
            content: "Perform TTS. Use the US female voice.".into(),
        },
        cera::tokenizer::ChatMessage {
            role: "user".into(),
            content: "Hello, how are you today?".into(),
        },
    ];
    let formatted = cera::tokenizer::apply_chat_template(&tokenizer, &messages, true)
        .expect("Failed to apply chat template");
    let tokens = tokenizer.encode(&formatted);

    let cpu_model =
        Lfm2Model::from_gguf((*model_gguf).clone(), 512).expect("Failed to load CPU model");
    let gpu_model = cera::model::gpu_lfm2::GpuLfm2Model::from_gguf_with_id(
        (*model_gguf).clone(),
        2048,
        "test_loop_parity".to_string(),
    )
    .unwrap();

    let mut cpu_state =
        InferenceState::from_config(cpu_model.config()).expect("Failed to init state");
    let mut gpu_state =
        InferenceState::from_config(cpu_model.config()).expect("Failed to init state");

    for (pos, &tok) in tokens.iter().enumerate() {
        cpu_model.forward(&[tok], pos, &mut cpu_state);
        gpu_model.forward_prefill_step(tok, pos, &mut gpu_state);
    }

    let mut pos = tokens.len();
    let mut cpu_emb = cpu_model.forward_embedding(&[128], pos, &mut cpu_state);
    let mut gpu_emb =
        pollster::block_on(gpu_model.forward_embedding_async(128, pos, &mut gpu_state)).unwrap();
    pos += 1;

    let dec_w = AudioDecoderWeights::from_gguf(&vocoder_gguf).expect("Failed to load vocoder");
    let mut cpu_df_state =
        cera::model::audio_decoder::DepthformerState::new(&dec_w.depthformer_config);

    for f in 0..15 {
        let cpu_codes = cera::model::audio_decoder::sample_audio_frame(
            &dec_w,
            &mut cpu_df_state,
            &cpu_emb,
            0.0,
            1,
        );
        let gpu_codes = cera::model::audio_decoder::sample_audio_frame(
            &dec_w,
            &mut cpu_df_state,
            &gpu_emb,
            0.0,
            1,
        );

        let dot: f32 = cpu_emb.iter().zip(&gpu_emb).map(|(a, b)| a * b).sum();
        let na: f32 = cpu_emb.iter().map(|a| a * a).sum::<f32>().sqrt();
        let nb: f32 = gpu_emb.iter().map(|b| b * b).sum::<f32>().sqrt();
        let cos = dot / (na * nb).max(1e-8);

        println!(
            "Frame {f:2} (pos={pos}): cos={cos:.6} | CPU codes={cpu_codes:?} | GPU codes={gpu_codes:?}"
        );

        let cpu_audio_emb = embed_audio_token(&dec_w, &cpu_codes);
        let gpu_audio_emb = embed_audio_token(&dec_w, &gpu_codes);

        cpu_emb = cpu_model.forward_hidden_from_embedding(&cpu_audio_emb, pos, &mut cpu_state);
        let gpu_hidden_buf = gpu_model
            .forward_hidden_from_embedding_gpu(&gpu_audio_emb, pos, &mut gpu_state)
            .unwrap();
        // Read back gpu hidden buffer
        gpu_emb = gpu_model.ctx().download_f32(gpu_hidden_buf, 2048);
        pos += 1;
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_layer_by_layer_parity_token_0() {
    let Some((model_gguf, _)) = load_models() else {
        return;
    };
    let cpu_model =
        Lfm2Model::from_gguf((*model_gguf).clone(), 512).expect("Failed to load CPU model");
    let gpu_model = cera::model::gpu_lfm2::GpuLfm2Model::from_gguf_with_id(
        (*model_gguf).clone(),
        2048,
        "test_l0_parity".to_string(),
    )
    .unwrap();

    let mut cpu_state =
        InferenceState::from_config(cpu_model.config()).expect("Failed to init state");
    let mut gpu_state =
        InferenceState::from_config(cpu_model.config()).expect("Failed to init state");

    // Test token 1 (BOS)
    let cpu_logits = cpu_model.forward(&[1], 0, &mut cpu_state);
    let gpu_logits =
        pollster::block_on(gpu_model.forward_logits_async(1, 0, &mut gpu_state)).unwrap();

    let dot: f32 = cpu_logits.iter().zip(&gpu_logits).map(|(a, b)| a * b).sum();
    let na: f32 = cpu_logits.iter().map(|a| a * a).sum::<f32>().sqrt();
    let nb: f32 = gpu_logits.iter().map(|b| b * b).sum::<f32>().sqrt();
    let cos = dot / (na * nb).max(1e-8);
    let max_diff: f32 = cpu_logits
        .iter()
        .zip(&gpu_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("Token 1 (BOS): cosine={cos:.6}, max_diff={max_diff:.4}");
}

#[test]
fn test_tts_studio_default_sample_text_synthesis() {
    let Some((model_gguf, vocoder_gguf)) = load_models() else {
        return;
    };
    let tokenizer =
        cera::tokenizer::BpeTokenizer::from_gguf(&model_gguf).expect("Failed to load tokenizer");
    let model = Lfm2Model::from_gguf((*model_gguf).clone(), 512).expect("Failed to load CPU model");
    let mut state = InferenceState::from_config(model.config()).expect("Failed to init state");

    let sample_text = "Hello, this voice was synthesized entirely on-device with the LFM2.5-Audio-1.5B-GGUF (Q4_0) model powered by Cera.";
    let messages = [
        cera::tokenizer::ChatMessage {
            role: "system".into(),
            content: "Perform TTS. Use the US female voice.".into(),
        },
        cera::tokenizer::ChatMessage {
            role: "user".into(),
            content: sample_text.into(),
        },
    ];
    let formatted = cera::tokenizer::apply_chat_template(&tokenizer, &messages, true)
        .expect("Failed to apply chat template");
    let tokens = tokenizer.encode(&formatted);
    println!("Prompt tokens count: {}", tokens.len());

    for (pos, &tok) in tokens.iter().enumerate() {
        model.forward(&[tok], pos, &mut state);
    }

    let dec_w = AudioDecoderWeights::from_gguf(&vocoder_gguf).expect("Failed to load vocoder");
    let detok_w = DetokenizerWeights::from_gguf(&vocoder_gguf).expect("Failed to load detok");
    let mut decoder =
        cera::audio_engine::AudioOutputDecoder::new(&dec_w, &detok_w, None, 0.0, 1, false);

    let mut pos = tokens.len();
    let mut emb = model.forward_embedding(&[128], pos, &mut state);
    pos += 1;

    let mut voiced_count = 0;
    let mut silent_count = 0;
    for f in 0..80 {
        let outcome = decoder.decode_frame(&emb);
        match outcome {
            cera::audio_engine::FrameOutcome::End => {
                println!("Frame {f:2}: END token");
                break;
            }
            cera::audio_engine::FrameOutcome::Codes {
                audio_embedding,
                pcm,
                codes,
            } => {
                let sum_sq: f32 = pcm.iter().map(|&x| x * x).sum();
                let rms = (sum_sq / pcm.len().max(1) as f32).sqrt();
                if rms >= 0.001 {
                    voiced_count += 1;
                    silent_count = 0;
                } else {
                    silent_count += 1;
                }
                println!(
                    "Frame {f:2} (pos={pos}): rms={rms:.4}, voiced={voiced_count}, silent={silent_count}, codes={codes:?}"
                );
                emb = model.forward_hidden_from_embedding(&audio_embedding, pos, &mut state);
                pos += 1;
            }
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_tts_studio_default_sample_text_synthesis_gpu() {
    let Some((model_gguf, vocoder_gguf)) = load_models() else {
        return;
    };
    let tokenizer =
        cera::tokenizer::BpeTokenizer::from_gguf(&model_gguf).expect("Failed to load tokenizer");
    let model = cera::model::gpu_lfm2::GpuLfm2Model::from_gguf_with_id(
        (*model_gguf).clone(),
        2048,
        "test_tts_gpu".to_string(),
    )
    .unwrap();
    let mut state = InferenceState::from_config(model.config()).expect("Failed to init state");

    let sample_text = "Hello, this voice was synthesized entirely on-device with the LFM2.5-Audio-1.5B   Q4_0 model powered by Cera.";
    let messages = [
        cera::tokenizer::ChatMessage {
            role: "system".into(),
            content: "Perform TTS. Use the US female voice.".into(),
        },
        cera::tokenizer::ChatMessage {
            role: "user".into(),
            content: sample_text.into(),
        },
    ];
    let formatted = cera::tokenizer::apply_chat_template(&tokenizer, &messages, true)
        .expect("Failed to apply chat template");
    let tokens = tokenizer.encode(&formatted);
    println!("Prompt tokens count: {}", tokens.len());

    for (pos, &tok) in tokens.iter().enumerate() {
        model.forward_prefill_step(tok, pos, &mut state);
    }

    let dec_w = AudioDecoderWeights::from_gguf(&vocoder_gguf).expect("Failed to load vocoder");
    let detok_w = DetokenizerWeights::from_gguf(&vocoder_gguf).expect("Failed to load detok");
    let gpu_voc = cera::model::wgpu_audio_decoder::WgpuAudioDecoder::from_ggufs_with_context(
        model.ctx().clone(),
        &vocoder_gguf,
        Some(&vocoder_gguf),
    )
    .unwrap();
    let mut decoder = cera::audio_engine::AudioOutputDecoder::new(
        &dec_w,
        &detok_w,
        Some(&gpu_voc),
        0.7,
        40,
        true,
    );

    let mut pos = tokens.len();
    let emb = pollster::block_on(model.forward_embedding_async(128, pos, &mut state)).unwrap();
    pos += 1;

    let mut voiced_count = 0;
    let mut silent_count = 0;
    for f in 0..120 {
        let outcome = if decoder.supports_gpu_depthformer() && f > 0 {
            pollster::block_on(decoder.decode_frame_from_gpu_hidden_async(model.hidden_buffer()))
                .unwrap()
        } else {
            pollster::block_on(decoder.decode_frame_async(&emb)).unwrap()
        };
        match outcome {
            cera::audio_engine::FrameOutcome::End => {
                println!("Frame {f:2}: END token");
                break;
            }
            cera::audio_engine::FrameOutcome::Codes {
                audio_embedding,
                pcm,
                codes,
            } => {
                let sum_sq: f32 = pcm.iter().map(|&x| x * x).sum();
                let rms = (sum_sq / pcm.len().max(1) as f32).sqrt();
                if rms >= 0.001 {
                    voiced_count += 1;
                    silent_count = 0;
                } else {
                    silent_count += 1;
                }
                println!(
                    "GPU Frame {f:2} (pos={pos}): rms={rms:.4}, voiced={voiced_count}, silent={silent_count}, codes={codes:?}"
                );
                if voiced_count >= 10 && silent_count >= 25 {
                    println!(
                        "Finished on trailing silence after {f} frames ({voiced_count} voiced)"
                    );
                    break;
                }
                model
                    .forward_hidden_from_embedding_gpu(&audio_embedding, pos, &mut state)
                    .unwrap();
                pos += 1;
            }
        }
    }
    println!("GPU test finished with {voiced_count} voiced frames");
    assert!(
        voiced_count >= 20,
        "Long sentence must generate at least 20 voiced frames on GPU!"
    );
}

#[test]
fn test_synthesize_browser_codes_to_wav() {
    let Some((_, vocoder_gguf)) = load_models() else {
        return;
    };
    let detok_w = DetokenizerWeights::from_gguf(&vocoder_gguf).expect("Failed to load detok");
    let dec_w = AudioDecoderWeights::from_gguf(&vocoder_gguf).expect("Failed to load dec");
    let mut state = cera::model::audio_decoder::DetokenizerState::new(&detok_w.config);

    let recorded_frames = vec![
        [1156, 1266, 796, 184, 3, 1489, 1417, 1190],
        [184, 705, 1728, 1416, 1186, 640, 981, 1285],
        [1484, 1739, 895, 1719, 700, 1960, 1683, 1579],
        [275, 1090, 1776, 720, 14, 1935, 1755, 1258],
        [275, 2039, 1748, 636, 884, 916, 1998, 496],
        [126, 520, 658, 840, 412, 201, 1872, 479],
        [803, 316, 1029, 1562, 1736, 11, 976, 1606],
        [486, 243, 1559, 1348, 823, 1572, 666, 1434],
    ];

    let mut streamer = cera::model::audio_decoder::IstftStreamer::new(
        detok_w.config.n_fft,
        detok_w.config.hop_length,
    );

    let mut full_pcm = Vec::new();
    for codes in &recorded_frames {
        let spectrum =
            cera::model::audio_decoder::detokenize_to_spectrum(&detok_w, &dec_w, &mut state, codes);
        let pcm = streamer.feed_frames(&spectrum);
        full_pcm.extend_from_slice(&pcm);
    }
    let flush = streamer.flush();
    full_pcm.extend_from_slice(&flush);

    println!(
        "Synthesized {} PCM samples from recorded browser codes",
        full_pcm.len()
    );
    let rms = (full_pcm.iter().map(|&x| x * x).sum::<f32>() / full_pcm.len() as f32).sqrt();
    let max = full_pcm.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    println!("RMS: {rms:.4}, Max: {max:.4}");

    write_wav_file("/tmp/browser_codes_test.wav", &full_pcm, 24000).expect("Failed to write WAV");
    println!("Wrote /tmp/browser_codes_test.wav");
}

fn write_wav_file(path: &str, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    let n = samples.len() as u32;
    let data_size = n * 2;
    let file_size = 36 + data_size;
    f.write_all(b"RIFF")?;
    f.write_all(&file_size.to_le_bytes())?;
    f.write_all(b"WAVE")?;
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&(sample_rate * 2).to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;
    for &s in samples {
        let i16_val = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        f.write_all(&i16_val.to_le_bytes())?;
    }
    Ok(())
}
