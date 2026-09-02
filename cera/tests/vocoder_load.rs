#[test]
fn load_vocoder_gguf() {
    let path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !path.exists() {
        eprintln!("skipping: vocoder not found");
        return;
    }

    let gguf = cera::gguf::GgufFile::open_arc(&path).unwrap();
    let weights = cera::model::audio_decoder::AudioDecoderWeights::from_gguf(&gguf).unwrap();

    let dc = &weights.depthformer_config;
    eprintln!(
        "Depthformer: {}L, embd={}, head={}, kv={}, hd={}, ffn={}",
        dc.n_layer, dc.n_embd, dc.n_head, dc.n_head_kv, dc.n_embd_head, dc.ffn_dim
    );

    let dec = &weights.decoder_config;
    eprintln!(
        "Decoder: {}cb, vocab={}, embd={}",
        dec.n_codebook, dec.n_vocab, dec.n_embd
    );

    assert_eq!(dc.n_layer, 6);
    assert_eq!(dc.n_embd, 1024);
    assert_eq!(dc.n_embd_head, 32);
    assert_eq!(dec.n_codebook, 8);
    assert_eq!(dec.n_vocab, 2049);
    assert_eq!(weights.depthformer_layers.len(), 6);
    assert_eq!(weights.depth_embeddings.len(), 8);
    eprintln!("Vocoder load OK");
}

#[test]
fn load_detokenizer() {
    let path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !path.exists() {
        eprintln!("skipping: vocoder not found");
        return;
    }

    let gguf = cera::gguf::GgufFile::open_arc(&path).unwrap();
    let detok = cera::model::audio_decoder::DetokenizerWeights::from_gguf(&gguf).unwrap();

    let c = &detok.config;
    eprintln!(
        "Detokenizer: {}L, embd={}, head={}/{}, ffn={}, n_fft={}, sr={}",
        c.n_layer, c.n_embd, c.n_head, c.n_head_kv, c.ffn_dim, c.n_fft, c.sample_rate
    );

    assert_eq!(c.n_layer, 8);
    assert_eq!(c.n_embd, 512);
    assert_eq!(c.n_head, 16);
    assert_eq!(c.n_head_kv, 8);
    assert_eq!(c.n_embd_head, 32);
    assert_eq!(detok.layers.len(), 8);
    assert_eq!(detok.lin_b.len(), 1282); // n_fft/2 + 1 = 641, × 2 (real/imag) = 1282
    eprintln!("Detokenizer load OK");
}

#[test]
fn load_q8_0_vocoder() {
    let path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join("models/liquid-ci/vocoder-LFM2.5-Audio-1.5B-Q8_0.gguf");
    if !path.exists() {
        eprintln!("skipping: Q8_0 vocoder not found");
        return;
    }

    let gguf = cera::gguf::GgufFile::open_arc(&path).unwrap();
    let dc = cera::model::audio_decoder::AudioDecoderWeights::from_gguf(&gguf);
    match dc {
        Ok(_) => eprintln!("Q8_0 AudioDecoderWeights load OK"),
        Err(e) => eprintln!("Q8_0 AudioDecoderWeights load FAILED: {e:#}"),
    }

    let detok = cera::model::audio_decoder::DetokenizerWeights::from_gguf(&gguf);
    match detok {
        Ok(_) => eprintln!("Q8_0 Detokenizer load OK"),
        Err(e) => eprintln!("Q8_0 Detokenizer load FAILED: {e:#}"),
    }
}

#[test]
fn test_detok_from_bytes() {
    let path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !path.exists() {
        eprintln!("skipping: vocoder not found");
        return;
    }

    let bytes = std::fs::read(&path).unwrap();
    let arc_bytes: std::sync::Arc<[u8]> = std::sync::Arc::from(bytes.into_boxed_slice());
    let gguf = cera::gguf::GgufFile::from_bytes(arc_bytes).unwrap();
    let arc_gguf = std::sync::Arc::new(gguf);
    let dw = cera::model::audio_decoder::DetokenizerWeights::from_gguf(&arc_gguf);
    match dw {
        Ok(_) => eprintln!("test_detok_from_bytes: OK"),
        Err(e) => panic!("DetokenizerWeights::from_gguf failed on in-memory GGUF: {e:#}"),
    }
}

#[test]
fn load_hf_tokenizer_gguf() {
    let path = std::path::PathBuf::from("/tmp/tokenizer-Q4_0.gguf");
    if !path.exists() {
        eprintln!("skipping: /tmp/tokenizer-Q4_0.gguf not found");
        return;
    }

    let gguf = cera::gguf::GgufFile::open_arc(&path).unwrap();
    // A standalone audio tokenizer GGUF has 65536 vocab tokens, not 16384 (8 * 2048).
    // The hardened DetokenizerWeights loader must reject it.
    let err = match cera::model::audio_decoder::DetokenizerWeights::from_gguf(&gguf) {
        Err(e) => e,
        Ok(_) => panic!("expected DetokenizerWeights::from_gguf to reject tokenizer GGUF"),
    };
    assert!(
        err.to_string().contains("emb_weight rows") || err.to_string().contains("mismatch"),
        "unexpected error message: {err:#}"
    );
    eprintln!("Hardened detokenizer rejection of audio tokenizer OK: {err:#}");
}

#[test]
fn test_lfm2_audio_prompt_special_token_encoding() {
    let path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !path.exists() {
        eprintln!("skipping: LFM2.5-Audio GGUF not found");
        return;
    }

    let gguf = cera::gguf::GgufFile::open(&path).unwrap();
    let tk = cera::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();

    let text = "<|startoftext|><|im_start|>system\nPerform TTS. Use the US female voice.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
    let ids = tk.encode_special(text, false);

    assert_eq!(ids.first(), Some(&1), "must start with <|startoftext|>");
    assert_eq!(ids.get(1), Some(&6), "must encode <|im_start|>");
    assert!(ids.contains(&7), "must encode <|im_end|>");
    assert!(ids.contains(&36309), "must encode 'Hello' token");
}

#[test]
fn test_audio_output_decoder_sequential_step() {
    let path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !path.exists() {
        eprintln!("skipping: vocoder GGUF not found");
        return;
    }

    let gguf = cera::gguf::GgufFile::open_arc(&path).unwrap();
    let decoder_weights =
        cera::model::audio_decoder::AudioDecoderWeights::from_gguf(&gguf).unwrap();
    let detok_weights = cera::model::audio_decoder::DetokenizerWeights::from_gguf(&gguf).unwrap();

    let mut decoder = cera::audio_engine::AudioOutputDecoder::new(
        &decoder_weights,
        &detok_weights,
        None,
        0.0,
        1,
        false,
    );

    // Feed a simulated hidden state of dim 2048
    let emb = vec![0.05f32; 2048];
    let outcome = decoder.decode_frame(&emb);
    match outcome {
        cera::audio_engine::FrameOutcome::End => {
            panic!("unexpected immediate End outcome for dummy embedding");
        }
        cera::audio_engine::FrameOutcome::Codes {
            audio_embedding,
            pcm,
            ..
        } => {
            assert_eq!(audio_embedding.len(), 2048);
            eprintln!(
                "Decoded frame 1: audio_embedding len={}, PCM len={}",
                audio_embedding.len(),
                pcm.len()
            );
        }
    }

    let mut finish_samples = 0;
    let samples = decoder.finish(&mut |pcm: &[f32], sr: u32| {
        finish_samples += pcm.len();
        assert_eq!(sr, 24000);
    });
    assert!(
        samples > 0 || finish_samples > 0,
        "must produce audio samples upon finish"
    );
}

#[test]
fn test_end_to_end_tts_synthesis() {
    let model_path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/LFM2.5-Audio-1.5B-Q4_0.gguf");
    let vocoder_path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !model_path.exists() || !vocoder_path.exists() {
        eprintln!("skipping: models not found");
        return;
    }

    let model_gguf = cera::gguf::GgufFile::open(&model_path).unwrap();
    let voc_gguf = cera::gguf::GgufFile::open_arc(&vocoder_path).unwrap();

    let tk = cera::tokenizer::BpeTokenizer::from_gguf(&model_gguf).unwrap();
    let model = cera::model::lfm2::Lfm2Model::from_gguf(model_gguf, 512).unwrap();
    let dec_w = cera::model::audio_decoder::AudioDecoderWeights::from_gguf(&voc_gguf).unwrap();
    let detok_w = cera::model::audio_decoder::DetokenizerWeights::from_gguf(&voc_gguf).unwrap();

    let prompt = "<|startoftext|><|im_start|>system\nPerform TTS. Use the US female voice.<|im_end|>\n<|im_start|>user\nHello, this voice was synthesized entirely on-device with the LFM2.5-Audio-1.5B model powered by Cera.<|im_end|>\n<|im_start|>assistant\n";
    let ids = tk.encode_special(prompt, false);

    use cera::model::Model;
    let mut state = cera::kv_cache::InferenceState::from_config(model.config()).unwrap();

    let mut logits = vec![];
    for (i, &tok) in ids.iter().enumerate() {
        logits = model.forward(&[tok], i, &mut state);
    }
    let next = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();

    eprintln!(
        "First token from prefill: {next} ({:?})",
        tk.decode(&[next])
    );

    let mut decoder =
        cera::audio_engine::AudioOutputDecoder::new(&dec_w, &detok_w, None, 0.0, 1, false);

    let mut pos = ids.len();
    let mut emb = model.forward_embedding(&[next], pos, &mut state);
    pos += 1;

    let mut all_pcm = vec![];
    for frame_idx in 0..150 {
        let outcome = decoder.decode_frame(&emb);
        match outcome {
            cera::audio_engine::FrameOutcome::End => {
                eprintln!("Frame {frame_idx}: End");
                break;
            }
            cera::audio_engine::FrameOutcome::Codes {
                audio_embedding,
                pcm,
                codes,
            } => {
                let sum_sq: f32 = pcm.iter().map(|&x| x * x).sum();
                let rms = (sum_sq / pcm.len().max(1) as f32).sqrt();
                if frame_idx < 10 || frame_idx % 20 == 0 {
                    eprintln!(
                        "Frame {frame_idx}: codes={codes:?}, pcm_len={}, rms={rms:.4}",
                        pcm.len()
                    );
                }
                all_pcm.extend_from_slice(&pcm);
                emb = model.forward_hidden_from_embedding(&audio_embedding, pos, &mut state);
                pos += 1;
            }
        }
    }

    decoder.finish(&mut |pcm: &[f32], _: u32| {
        all_pcm.extend_from_slice(pcm);
    });

    eprintln!("Total synthesized PCM samples: {}", all_pcm.len());
    if !all_pcm.is_empty() {
        let mut f = std::fs::File::create("/tmp/tts_cpu_hello.wav").unwrap();
        use std::io::Write;
        let n = all_pcm.len() as u32;
        let data_size = n * 2;
        let file_size = 36 + data_size;
        f.write_all(b"RIFF").unwrap();
        f.write_all(&file_size.to_le_bytes()).unwrap();
        f.write_all(b"WAVE").unwrap();
        f.write_all(b"fmt ").unwrap();
        f.write_all(&16u32.to_le_bytes()).unwrap();
        f.write_all(&1u16.to_le_bytes()).unwrap();
        f.write_all(&1u16.to_le_bytes()).unwrap();
        f.write_all(&24000u32.to_le_bytes()).unwrap();
        f.write_all(&(24000u32 * 2).to_le_bytes()).unwrap();
        f.write_all(&2u16.to_le_bytes()).unwrap();
        f.write_all(&16u16.to_le_bytes()).unwrap();
        f.write_all(b"data").unwrap();
        f.write_all(&data_size.to_le_bytes()).unwrap();
        for &s in &all_pcm {
            let i16_val = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
            f.write_all(&i16_val.to_le_bytes()).unwrap();
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_end_to_end_tts_synthesis_gpu_parity() {
    let base_path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0");
    let model_path = base_path.join("LFM2.5-Audio-1.5B-Q4_0.gguf");
    let vocoder_path = base_path.join("vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf");

    if !model_path.exists() || !vocoder_path.exists() {
        eprintln!("model or vocoder not found, skipping GPU parity test");
        return;
    }

    let gguf = cera::gguf::GgufFile::open_arc(&model_path).unwrap();
    let voc_gguf = cera::gguf::GgufFile::open_arc(&vocoder_path).unwrap();
    let tk = cera::tokenizer::BpeTokenizer::from_gguf(&gguf).unwrap();
    let dec_w = cera::model::audio_decoder::AudioDecoderWeights::from_gguf(&voc_gguf).unwrap();
    let detok_w = cera::model::audio_decoder::DetokenizerWeights::from_gguf(&voc_gguf).unwrap();
    let cpu_model = cera::model::lfm2::Lfm2Model::from_gguf((*gguf).clone(), 2048).unwrap();
    let gpu_model = cera::model::gpu_lfm2::GpuLfm2Model::from_gguf_with_id(
        (*gguf).clone(),
        2048,
        "test_gpu".to_string(),
    )
    .unwrap();

    let gpu_voc = cera::model::wgpu_audio_decoder::WgpuAudioDecoder::from_ggufs_with_context(
        gpu_model.ctx().clone(),
        &voc_gguf,
        Some(&voc_gguf),
    )
    .unwrap();

    let prompt = "<|startoftext|><|im_start|>system\nPerform TTS. Use the US female voice.<|im_end|>\n<|im_start|>user\nHello, this voice was synthesized entirely on-device with the LFM2.5-Audio-1.5B model powered by Cera.<|im_end|>\n<|im_start|>assistant\n";
    let ids = tk.encode_special(prompt, false);

    use cera::model::Model;
    let mut cpu_state = cera::kv_cache::InferenceState::from_config(cpu_model.config()).unwrap();
    let mut gpu_state = cera::kv_cache::InferenceState::from_config(gpu_model.config()).unwrap();

    let mut cpu_logits = vec![];
    let mut gpu_logits = vec![];
    for (i, &tok) in ids.iter().enumerate() {
        cpu_logits = cpu_model.forward(&[tok], i, &mut cpu_state);
        gpu_logits = gpu_model.forward(&[tok], i, &mut gpu_state);
        let dot: f32 = cpu_logits.iter().zip(&gpu_logits).map(|(a, b)| a * b).sum();
        let na: f32 = cpu_logits.iter().map(|a| a * a).sum::<f32>().sqrt();
        let nb: f32 = gpu_logits.iter().map(|b| b * b).sum::<f32>().sqrt();
        let cos = dot / (na * nb).max(1e-8);
        if i < 5 || i % 10 == 0 || i == ids.len() - 1 {
            eprintln!("Prefill step {i} (token {tok}): logit_cosine={cos:.6}");
        }
    }
    let cpu_next = cpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();
    let gpu_next = gpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();

    eprintln!("First token from prefill: CPU={cpu_next}, GPU={gpu_next}");
    assert_eq!(cpu_next, gpu_next);

    let mut cpu_decoder =
        cera::audio_engine::AudioOutputDecoder::new(&dec_w, &detok_w, None, 0.0, 1, false);
    let mut gpu_decoder = cera::audio_engine::AudioOutputDecoder::new(
        &dec_w,
        &detok_w,
        Some(&gpu_voc),
        0.0,
        1,
        false,
    );

    let mut pos = ids.len();
    let mut cpu_emb = cpu_model.forward_embedding(&[cpu_next], pos, &mut cpu_state);
    let gpu_emb_0 =
        pollster::block_on(gpu_model.forward_embedding_async(gpu_next, pos, &mut gpu_state))
            .unwrap();
    let cos_0 = {
        let dot: f32 = cpu_emb.iter().zip(&gpu_emb_0).map(|(a, b)| a * b).sum();
        let na: f32 = cpu_emb.iter().map(|a| a * a).sum::<f32>().sqrt();
        let nb: f32 = gpu_emb_0.iter().map(|b| b * b).sum::<f32>().sqrt();
        dot / (na * nb).max(1e-8)
    };
    eprintln!("Initial audio_start hidden state cosine: {cos_0:.6}");
    pos += 1;

    let mut cpu_all_pcm = vec![];
    let mut gpu_all_pcm = vec![];
    let mut gpu_emb = gpu_emb_0;
    for frame_idx in 0..100 {
        let cpu_outcome = cpu_decoder.decode_frame(&cpu_emb);
        let gpu_outcome = gpu_decoder.decode_frame(&gpu_emb);

        match (cpu_outcome, gpu_outcome) {
            (
                cera::audio_engine::FrameOutcome::Codes {
                    audio_embedding: cpu_ae,
                    codes: cpu_c,
                    pcm: cpu_pcm,
                },
                cera::audio_engine::FrameOutcome::Codes {
                    audio_embedding: gpu_ae,
                    codes: gpu_c,
                    pcm: gpu_pcm,
                },
            ) => {
                let sum_sq: f32 = gpu_pcm.iter().map(|&x| x * x).sum();
                let rms = (sum_sq / gpu_pcm.len().max(1) as f32).sqrt();
                let matching = cpu_c.iter().zip(&gpu_c).filter(|(a, b)| a == b).count();
                cpu_all_pcm.extend_from_slice(&cpu_pcm);
                gpu_all_pcm.extend_from_slice(&gpu_pcm);

                cpu_emb = cpu_model.forward_hidden_from_embedding(&cpu_ae, pos, &mut cpu_state);
                gpu_emb = pollster::block_on(gpu_model.forward_hidden_from_embedding_async(
                    &gpu_ae,
                    pos,
                    &mut gpu_state,
                ))
                .unwrap();
                let cos_f = {
                    let dot: f32 = cpu_emb.iter().zip(&gpu_emb).map(|(a, b)| a * b).sum();
                    let na: f32 = cpu_emb.iter().map(|a| a * a).sum::<f32>().sqrt();
                    let nb: f32 = gpu_emb.iter().map(|b| b * b).sum::<f32>().sqrt();
                    dot / (na * nb).max(1e-8)
                };
                if frame_idx < 15 || frame_idx % 20 == 0 {
                    eprintln!(
                        "Frame {frame_idx}: cos={cos_f:.6}, matching={matching}/8, gpu_rms={rms:.4}\n  CPU codes={cpu_c:?}\n  GPU codes={gpu_c:?}"
                    );
                }
                pos += 1;
            }
            _ => break,
        }
    }

    cpu_decoder.finish(&mut |pcm: &[f32], _: u32| {
        cpu_all_pcm.extend_from_slice(pcm);
    });
    gpu_decoder.finish(&mut |pcm: &[f32], _: u32| {
        gpu_all_pcm.extend_from_slice(pcm);
    });

    eprintln!(
        "Total CPU PCM samples: {}, GPU: {}",
        cpu_all_pcm.len(),
        gpu_all_pcm.len()
    );
    let write_wav = |path: &str, pcm: &[f32]| {
        use std::io::Write;
        let mut f = std::fs::File::create(path).unwrap();
        let n = pcm.len() as u32;
        let data_size = n * 2;
        let file_size = 36 + data_size;
        f.write_all(b"RIFF").unwrap();
        f.write_all(&file_size.to_le_bytes()).unwrap();
        f.write_all(b"WAVE").unwrap();
        f.write_all(b"fmt ").unwrap();
        f.write_all(&16u32.to_le_bytes()).unwrap();
        f.write_all(&1u16.to_le_bytes()).unwrap();
        f.write_all(&1u16.to_le_bytes()).unwrap();
        f.write_all(&24000u32.to_le_bytes()).unwrap();
        f.write_all(&(24000u32 * 2).to_le_bytes()).unwrap();
        f.write_all(&2u16.to_le_bytes()).unwrap();
        f.write_all(&16u16.to_le_bytes()).unwrap();
        f.write_all(b"data").unwrap();
        f.write_all(&data_size.to_le_bytes()).unwrap();
        for &s in pcm {
            let i16_val = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
            f.write_all(&i16_val.to_le_bytes()).unwrap();
        }
    };
    write_wav("/tmp/tts_cpu_hello.wav", &cpu_all_pcm);
    write_wav("/tmp/tts_gpu_hello.wav", &gpu_all_pcm);
}
