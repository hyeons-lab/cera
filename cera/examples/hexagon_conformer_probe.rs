//! Check the Hexagon Conformer encoder, stage by stage, against the CPU
//! encoder on a device.
//!
//! The NPU encoder (`model::audio_encoder_hexagon`) is built in milestones,
//! and each one is only trusted once it agrees with `model::audio_encoder` on
//! real activations: the input here is the CPU conv stem's output for a
//! synthetic speech-like clip, and every stage starts from the same input on
//! both sides, so an error shows up in the stage that made it.
//!
//! ```text
//! cargo ndk -t arm64-v8a build --release -p cera --example hexagon_conformer_probe --features hexagon
//! adb push target/aarch64-linux-android/release/examples/hexagon_conformer_probe /data/local/tmp/cera-bench/
//! adb shell 'cd /data/local/tmp/cera-bench && echo 0 > /proc/$$/oom_score_adj && \
//!   ./hexagon_conformer_probe audio/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf 6'
//! ```
//!
//! Arguments: the mmproj GGUF and the clip length in seconds (default 6).

/// Prints how one stage's NPU output differs from the CPU's.
#[cfg(feature = "hexagon")]
type Report<'a> = &'a dyn Fn(&str, &[f32], &[f32]);

/// Compare the attention's intermediates with a straightforward F64 CPU
/// recomputation from the NPU's own inputs to each stage, so the first stage
/// that differs is the one with the bug.
#[cfg(feature = "hexagon")]
fn attention_stages(
    enc: &cera::model::audio_encoder_hexagon::HexagonAudioEncoder,
    weights: &cera::model::audio_encoder::AudioEncoderWeights,
    x0: &[f32],
    t: usize,
    layer: usize,
    pos: &[f32],
    report: Report<'_>,
) {
    let cfg = &weights.config;
    let l = &weights.layers[layer];
    let (n, h) = (cfg.n_embd, cfg.n_head);
    let d = n / h;
    let dump = enc.debug_attention(layer, x0, t).expect("attention dump");
    let tag = |s: &str| format!("L{layer} attn {s}");

    // Q, K, V from LayerNorm(x), like the CPU path.
    let (mut q, mut k, mut v) = (vec![0f32; t * n], vec![0f32; t * n], vec![0f32; t * n]);
    for ti in 0..t {
        let mut pre = x0[ti * n..(ti + 1) * n].to_vec();
        cera::backend::cpu::layer_norm_inplace(&mut pre, &l.ln1_w, &l.ln1_b, cfg.eps);
        for (w, b, out) in [
            (&l.attn_q_w, &l.attn_q_b, &mut q),
            (&l.attn_k_w, &l.attn_k_b, &mut k),
            (&l.attn_v_w, &l.attn_v_b, &mut v),
        ] {
            let row = &mut out[ti * n..(ti + 1) * n];
            w.gemv(&pre, row);
            cera::backend::cpu::add_inplace(row, b);
        }
    }
    report(&tag("q"), &q, &dump.q);
    report(&tag("k"), &k, &dump.k);
    report(&tag("v"), &v, &dump.v);

    let lp = 2 * t - 1;
    let mut p = vec![0f32; lp * n];
    for pi in 0..lp {
        l.linear_pos_w
            .gemv(&pos[pi * 512..(pi + 1) * 512], &mut p[pi * n..(pi + 1) * n]);
    }
    report(&tag("p"), &p, &dump.p);

    let scale = 1.0 / (d as f64).sqrt();
    let (mut qu, mut qv) = (q.clone(), q.clone());
    for ti in 0..t {
        for c in 0..n {
            qu[ti * n + c] += l.pos_bias_u[c];
            qv[ti * n + c] = ((q[ti * n + c] + l.pos_bias_v[c]) as f64 * scale) as f32;
        }
    }
    report(&tag("qu"), &qu, &dump.qu);
    report(&tag("qv*s"), &qv, &dump.qv);

    // Scores from the NPU's own K, Q+u, P, (Q+v)*s so each is judged alone.
    let dot = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| *x as f64 * *y as f64)
            .sum::<f64>()
    };
    let (mut ac, mut bd) = (vec![0f32; h * t * t], vec![0f32; h * t * lp]);
    for hh in 0..h {
        for qi in 0..t {
            let qu_h = &dump.qu[qi * n + hh * d..qi * n + (hh + 1) * d];
            let qv_h = &dump.qv[qi * n + hh * d..qi * n + (hh + 1) * d];
            for ki in 0..t {
                ac[(hh * t + qi) * t + ki] =
                    dot(qu_h, &dump.k[ki * n + hh * d..ki * n + (hh + 1) * d]) as f32;
            }
            for j in 0..lp {
                bd[(hh * t + qi) * lp + j] =
                    dot(qv_h, &dump.p[j * n + hh * d..j * n + (hh + 1) * d]) as f32;
            }
        }
    }
    report(&tag("ac (content scores)"), &ac, &dump.ac);
    report(&tag("bd (position scores)"), &bd, &dump.bd);

    // Softmax of scale*ac + shifted(bd) from the NPU's own scores.
    let mut sm = vec![0f32; h * t * t];
    for hh in 0..h {
        for qi in 0..t {
            let row: Vec<f64> = (0..t)
                .map(|ki| {
                    dump.ac[(hh * t + qi) * t + ki] as f64 * scale
                        + dump.bd[(hh * t + qi) * lp + (t - 1 - qi + ki)] as f64
                })
                .collect();
            let m = row.iter().cloned().fold(f64::MIN, f64::max);
            let sum: f64 = row.iter().map(|x| (x - m).exp()).sum();
            for ki in 0..t {
                sm[(hh * t + qi) * t + ki] = ((row[ki] - m).exp() / sum) as f32;
            }
        }
    }
    report(&tag("softmax"), &sm, &dump.sm);

    // V transposed per head, then attn @ V from the NPU's softmax and V.
    let mut vt = vec![0f32; h * d * t];
    for hh in 0..h {
        for dd in 0..d {
            for ki in 0..t {
                vt[(hh * d + dd) * t + ki] = dump.v[ki * n + hh * d + dd];
            }
        }
    }
    report(&tag("v^T"), &vt, &dump.vt);
    let mut av = vec![0f32; t * n];
    for hh in 0..h {
        for qi in 0..t {
            for dd in 0..d {
                let mut acc = 0f64;
                for ki in 0..t {
                    acc += dump.sm[(hh * t + qi) * t + ki] as f64
                        * dump.v[ki * n + hh * d + dd] as f64;
                }
                av[qi * n + hh * d + dd] = acc as f32;
            }
        }
    }
    report(&tag("attn@V"), &av, &dump.av);
}

#[cfg(feature = "hexagon")]
fn main() {
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use cera::backend::hexagon::{FastRpcDriver, probe_device};
    use cera::gguf::GgufFile;
    use cera::model::audio_encoder::{
        AudioEncoderWeights, SAMPLE_RATE, conformer_conv_module_forward, conformer_ffn_forward,
        conformer_self_attention_forward, conv_stem_forward, relative_pos_emb,
    };
    use cera::model::audio_encoder_hexagon::HexagonAudioEncoder;
    use cera::model::audio_preprocessor::log_mel_spectrogram;

    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: hexagon_conformer_probe <mmproj.gguf> [seconds]");
    let seconds: f64 = args.next().map_or(6.0, |s| s.parse().expect("seconds"));

    let gguf = GgufFile::open_arc(Path::new(&path)).expect("open mmproj");
    let weights = Arc::new(AudioEncoderWeights::from_gguf(&gguf).expect("audio encoder weights"));
    let cfg = weights.config.clone();

    // A speech-like clip: harmonics with a slow amplitude envelope, plus noise.
    let n = (seconds * SAMPLE_RATE as f64) as usize;
    let mut seed = 0x9E3779B9u32;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5;
            let t = i as f32 / SAMPLE_RATE as f32;
            let env = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * 2.3 * t).sin();
            env * (0.3 * (2.0 * std::f32::consts::PI * 150.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 450.0 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 1350.0 * t).sin())
                + 0.02 * noise
        })
        .collect();
    let t_mel = std::time::Instant::now();
    let (mel, n_frames) = log_mel_spectrogram(&pcm, cfg.n_mel_bins);
    let mel_ms = t_mel.elapsed().as_secs_f64() * 1e3;
    // Where the CPU stem's time goes, layer by layer (same loop as
    // `conv_stem_forward`, timed).
    {
        let modes = [
            (0usize, 2usize, 1usize),
            (1, 2, 1),
            (0, 1, 0),
            (1, 2, 1),
            (0, 1, 0),
        ];
        let relu = [true, false, true, false, true];
        let (mut cur, mut ch, mut h, mut w) = (mel.clone(), 1usize, n_frames, cfg.n_mel_bins);
        for (i, layer) in weights.conv_stem.layers.iter().enumerate() {
            let (kw, kh, ipg, och) = (
                layer.shape[0],
                layer.shape[1],
                layer.shape[2],
                layer.shape[3],
            );
            let (kind, stride, pad) = modes[i];
            let groups = if kind == 0 { 1 } else { ch };
            let (nh, nw) = (
                (h + 2 * pad - kh) / stride + 1,
                (w + 2 * pad - kw) / stride + 1,
            );
            let mut next = vec![0.0f32; och * nh * nw];
            let t0 = std::time::Instant::now();
            cera::backend::cpu::conv2d(
                &cur,
                &layer.weight,
                Some(&layer.bias),
                &mut next,
                ch,
                och,
                h,
                w,
                kh,
                kw,
                stride,
                stride,
                pad,
                pad,
                groups,
            );
            if relu[i] {
                cera::backend::cpu::relu_inplace(&mut next);
            }
            println!(
                "   stem layer {i}: {ch}x{h}x{w} -> {och}x{nh}x{nw} (groups {groups}, in/group {ipg}): {:.1} ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
            (cur, ch, h, w) = (next, och, nh, nw);
        }
    }
    let t_stem = std::time::Instant::now();
    let (x0, t) = conv_stem_forward(&mel, n_frames, &weights.conv_stem, &cfg);
    println!(
        "CPU front end for {seconds:.1} s: log-mel {mel_ms:.0} ms, conv stem {:.0} ms",
        t_stem.elapsed().as_secs_f64() * 1e3
    );
    println!("clip {seconds:.1} s -> {n_frames} mel frames -> {t} encoder frames");

    let driver = FastRpcDriver::load().expect("load the FastRPC driver");
    let device = probe_device(&driver, None).expect("open a DSP session");
    let enc = HexagonAudioEncoder::new(
        driver,
        Arc::new(Mutex::new(device)),
        &weights,
        cera::model::audio_encoder_hexagon::MAX_FRAMES,
    )
    .expect("stage the encoder");

    let report = |name: &str, cpu: &[f32], npu: &[f32]| {
        assert_eq!(cpu.len(), npu.len());
        let (mut max_abs, mut num, mut den, mut dot, mut nc, mut nn) =
            (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        for (&c, &g) in cpu.iter().zip(npu) {
            let (c, g) = (c as f64, g as f64);
            max_abs = max_abs.max((c - g).abs());
            num += (c - g) * (c - g);
            den += c * c;
            dot += c * g;
            nc += c * c;
            nn += g * g;
        }
        println!(
            "{name:<22} max|diff| {max_abs:.4e}  rel-rms {:.4e}  cosine {:.6}",
            (num / den.max(1e-30)).sqrt(),
            dot / (nc.sqrt() * nn.sqrt()).max(1e-30)
        );
    };

    for (layer, second) in [(0usize, false), (0, true), (1, false)] {
        let l = &weights.layers[layer];
        let (nw, nb, up_w, up_b, down_w, down_b) = if second {
            (
                &l.ffn_norm_1_w,
                &l.ffn_norm_1_b,
                &l.ffn_up_1_w,
                &l.ffn_up_1_b,
                &l.ffn_down_1_w,
                &l.ffn_down_1_b,
            )
        } else {
            (
                &l.ffn_norm_w,
                &l.ffn_norm_b,
                &l.ffn_up_w,
                &l.ffn_up_b,
                &l.ffn_down_w,
                &l.ffn_down_b,
            )
        };
        let mut cpu = x0.clone();
        let mut pre = vec![0.0f32; cfg.n_embd];
        let mut ff = vec![0.0f32; cfg.n_ff];
        conformer_ffn_forward(
            &mut cpu, nw, nb, up_w, up_b, down_w, down_b, cfg.n_embd, cfg.n_ff, t, cfg.eps,
            &mut pre, &mut ff,
        );
        let npu = enc.run_ffn(layer, second, &x0, t).expect("NPU ffn");
        report(
            &format!("layer {layer} ffn{}", if second { 2 } else { 1 }),
            &cpu,
            &npu,
        );
    }

    let pos = relative_pos_emb(t);

    // Whole blocks: the CPU encoder's block order, run for 1, 2 and all
    // blocks, against the same on the NPU. Errors compound with depth.
    let cpu_block = |x: &mut Vec<f32>, layer: usize| {
        let l = &weights.layers[layer];
        let mut pre = vec![0.0f32; cfg.n_embd];
        let mut ff = vec![0.0f32; cfg.n_ff];
        conformer_ffn_forward(
            x,
            &l.ffn_norm_w,
            &l.ffn_norm_b,
            &l.ffn_up_w,
            &l.ffn_up_b,
            &l.ffn_down_w,
            &l.ffn_down_b,
            cfg.n_embd,
            cfg.n_ff,
            t,
            cfg.eps,
            &mut pre,
            &mut ff,
        );
        conformer_self_attention_forward(
            x,
            &pos,
            &l.ln1_w,
            &l.ln1_b,
            &l.attn_q_w,
            &l.attn_q_b,
            &l.attn_k_w,
            &l.attn_k_b,
            &l.attn_v_w,
            &l.attn_v_b,
            &l.attn_o_w,
            &l.attn_o_b,
            &l.pos_bias_u,
            &l.pos_bias_v,
            &l.linear_pos_w,
            cfg.n_embd,
            cfg.n_head,
            t,
            cfg.eps,
        );
        conformer_conv_module_forward(
            x,
            &l.norm_conv_w,
            &l.norm_conv_b,
            &l.conv_pw1_w,
            &l.conv_pw1_b,
            &l.conv_dw_w,
            &l.conv_dw_b,
            &l.conv_norm_w,
            &l.conv_norm_b,
            &l.conv_pw2_w,
            &l.conv_pw2_b,
            cfg.n_embd,
            t,
            l.conv_dw_w.len() / cfg.n_embd,
            cfg.eps,
        );
        conformer_ffn_forward(
            x,
            &l.ffn_norm_1_w,
            &l.ffn_norm_1_b,
            &l.ffn_up_1_w,
            &l.ffn_up_1_b,
            &l.ffn_down_1_w,
            &l.ffn_down_1_b,
            cfg.n_embd,
            cfg.n_ff,
            t,
            cfg.eps,
            &mut pre,
            &mut ff,
        );
        for row in x.chunks_exact_mut(cfg.n_embd) {
            cera::backend::cpu::layer_norm_inplace(row, &l.ln2_w, &l.ln2_b, cfg.eps);
        }
    };
    let mut cpu = x0.clone();
    let n_layers = weights.layers.len();
    for done in 1..=n_layers {
        cpu_block(&mut cpu, done - 1);
        if [1, 2, 4, 8, n_layers].contains(&done) {
            let t0 = std::time::Instant::now();
            match enc.run_blocks(done, &x0, t) {
                Ok(npu) => {
                    report(&format!("{done} block(s)"), &cpu, &npu);
                    println!(
                        "   ({} blocks on the NPU: {:.0} ms)",
                        done,
                        t0.elapsed().as_secs_f64() * 1e3
                    );
                }
                Err(e) => println!("{done} block(s): NPU error: {e}"),
            }
        }
    }

    for layer in [0usize, 1, 8, 16] {
        let layer = layer.min(weights.layers.len() - 1);
        let l = &weights.layers[layer];
        let mut cpu = x0.clone();
        conformer_self_attention_forward(
            &mut cpu,
            &pos,
            &l.ln1_w,
            &l.ln1_b,
            &l.attn_q_w,
            &l.attn_q_b,
            &l.attn_k_w,
            &l.attn_k_b,
            &l.attn_v_w,
            &l.attn_v_b,
            &l.attn_o_w,
            &l.attn_o_b,
            &l.pos_bias_u,
            &l.pos_bias_v,
            &l.linear_pos_w,
            cfg.n_embd,
            cfg.n_head,
            t,
            cfg.eps,
        );
        if layer == 0 || layer == 8 {
            attention_stages(&enc, &weights, &x0, t, layer, &pos, &report);
        }
        match enc.run_attention(layer, &x0, t) {
            Ok(npu) => report(&format!("layer {layer} attention"), &cpu, &npu),
            Err(e) => println!("layer {layer} attention: NPU error: {e}"),
        }
    }

    for layer in [0usize, 1, 8, 16] {
        let l = &weights.layers[layer.min(weights.layers.len() - 1)];
        let layer = layer.min(weights.layers.len() - 1);
        let kernel = l.conv_dw_w.len() / cfg.n_embd;
        let mut cpu = x0.clone();
        conformer_conv_module_forward(
            &mut cpu,
            &l.norm_conv_w,
            &l.norm_conv_b,
            &l.conv_pw1_w,
            &l.conv_pw1_b,
            &l.conv_dw_w,
            &l.conv_dw_b,
            &l.conv_norm_w,
            &l.conv_norm_b,
            &l.conv_pw2_w,
            &l.conv_pw2_b,
            cfg.n_embd,
            t,
            kernel,
            cfg.eps,
        );
        let npu = enc.run_conv(layer, &x0, t).expect("NPU conv module");
        report(&format!("layer {layer} conv"), &cpu, &npu);
    }

    // The conv stem, stage by stage against the CPU's convolutions.
    {
        let (mel, n_frames) =
            cera::model::audio_preprocessor::log_mel_spectrogram(&pcm, weights.config.n_mel_bins);
        let layers = &weights.conv_stem.layers;
        // (depthwise, stride, pad, relu) per stem layer, as in conv_stem_forward.
        let modes = [
            (false, 2, 1, true),
            (true, 2, 1, false),
            (false, 1, 0, true),
            (true, 2, 1, false),
            (false, 1, 0, true),
        ];
        let (mut cur, mut c_in, mut h, mut w) =
            (mel.clone(), 1usize, n_frames, weights.config.n_mel_bins);
        let mut cpu = Vec::new();
        for (layer, &(depthwise, stride, pad, relu)) in layers.iter().zip(&modes) {
            let (kw, kh, out_ch) = (layer.shape[0], layer.shape[1], layer.shape[3]);
            let (nh, nw) = (
                (h + 2 * pad - kh) / stride + 1,
                (w + 2 * pad - kw) / stride + 1,
            );
            let mut next = vec![0f32; out_ch * nh * nw];
            let groups = if depthwise { c_in } else { 1 };
            cera::backend::cpu::conv2d(
                &cur,
                &layer.weight,
                Some(&layer.bias),
                &mut next,
                c_in,
                out_ch,
                h,
                w,
                kh,
                kw,
                stride,
                stride,
                pad,
                pad,
                groups,
            );
            if relu {
                cera::backend::cpu::relu_inplace(&mut next);
            }
            cpu.push(next.clone());
            (cur, c_in, h, w) = (next, out_ch, nh, nw);
        }
        // The projection's input, per time step: [channel, freq].
        let mut flat = vec![0f32; h * c_in * w];
        for ti in 0..h {
            for c in 0..c_in {
                for f in 0..w {
                    flat[ti * c_in * w + c * w + f] = cur[(c * h + ti) * w + f];
                }
            }
        }
        let (cpu_out, cpu_t) = cera::model::audio_encoder::conv_stem_forward(
            &mel,
            n_frames,
            &weights.conv_stem,
            &weights.config,
        );
        match enc.debug_stem(&mel, n_frames) {
            Ok(d) => {
                println!(
                    "conv stem on the NPU ({n_frames} mel frames -> {} frames):",
                    d.t
                );
                assert_eq!(d.t, cpu_t);
                report("stem conv 0 (3x3 s2, relu)", &cpu[0], &d.l0);
                report("stem conv 1 (dw 3x3 s2)", &cpu[1], &d.dw1);
                report("stem conv 2 (pw, relu)", &cpu[2], &d.pw2);
                report("stem conv 3 (dw 3x3 s2)", &cpu[3], &d.dw3);
                report("stem conv 4 (pw, relu)", &cpu[4], &d.pw4);
                report("stem flatten", &flat, &d.flat);
                report("stem output", &cpu_out, &d.out);
            }
            Err(e) => println!("conv stem: NPU error: {e}"),
        }
    }

    // End to end on PCM: CPU front end plus NPU blocks and adapter against the
    // all-CPU encoder, with what each costs the CPU.
    // (user, system) CPU seconds of this process.
    let cpu_split = || {
        let mut ru = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        // SAFETY: `getrusage` fills the struct; RUSAGE_SELF is valid.
        let ru = unsafe {
            libc::getrusage(libc::RUSAGE_SELF, ru.as_mut_ptr());
            ru.assume_init()
        };
        let secs = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
        (secs(ru.ru_utime), secs(ru.ru_stime))
    };
    let cpu_seconds = || {
        let (u, s) = cpu_split();
        u + s
    };
    let (cpu_emb, cpu_t) = cera::model::audio_encoder::encode_audio_pcm(&pcm, &weights);
    match enc.encode(&pcm) {
        Ok((npu_emb, npu_t)) => {
            assert_eq!(cpu_t, npu_t);
            report("end-to-end embeddings", &cpu_emb, &npu_emb);
        }
        Err(e) => println!("end-to-end: NPU error: {e}"),
    }
    let reps = 5;
    for (name, run) in [
        (
            "CPU encoder",
            &(|| drop(cera::model::audio_encoder::encode_audio_pcm(&pcm, &weights))) as &dyn Fn(),
        ),
        (
            "NPU encoder",
            &(|| drop(enc.encode(&pcm).expect("NPU encode"))),
        ),
    ] {
        run(); // warm-up
        let (c0, t0) = (cpu_seconds(), std::time::Instant::now());
        for _ in 0..reps {
            run();
        }
        let (wall, cpu) = (
            t0.elapsed().as_secs_f64() / reps as f64,
            (cpu_seconds() - c0) / reps as f64,
        );
        println!(
            "{name:<12} wall {:.0} ms ({:.3} s per audio s), cpu {:.0} ms ({:.3} cpu-s per audio s)",
            wall * 1e3,
            wall / seconds,
            cpu * 1e3,
            cpu / seconds
        );
    }

    // Where the NPU path's CPU time goes.
    let cfg = &weights.config;
    let measure = |name: &str, run: &mut dyn FnMut()| {
        run();
        let ((u0, s0), t0) = (cpu_split(), std::time::Instant::now());
        for _ in 0..reps {
            run();
        }
        let (u1, s1) = cpu_split();
        let per = |d: f64| d / reps as f64 * 1e3;
        println!(
            "  {name:<22} wall {:5.0} ms  cpu {:5.0} ms (user {:5.0}, sys {:5.0})",
            per(t0.elapsed().as_secs_f64()),
            per(u1 - u0 + s1 - s0),
            per(u1 - u0),
            per(s1 - s0)
        );
    };
    println!("NPU path phases:");
    let (mel, n_frames) =
        cera::model::audio_preprocessor::log_mel_spectrogram(&pcm, cfg.n_mel_bins);
    let npu_mel = enc.log_mel_npu(&pcm, n_frames).expect("NPU log-mel");
    report("log-mel on the NPU", &mel, &npu_mel);
    measure("log-mel (CPU)", &mut || {
        drop(cera::model::audio_preprocessor::log_mel_spectrogram(
            &pcm,
            cfg.n_mel_bins,
        ))
    });
    measure("log-mel (NPU)", &mut || {
        drop(enc.log_mel_npu(&pcm, n_frames).expect("NPU log-mel"))
    });
    measure("conv stem (CPU)", &mut || {
        drop(cera::model::audio_encoder::conv_stem_forward(
            &mel,
            n_frames,
            &weights.conv_stem,
            cfg,
        ))
    });
    measure("stem+blocks+adapter", &mut || {
        drop(enc.encode_mel(&mel, n_frames).expect("NPU encode"))
    });
}

#[cfg(not(feature = "hexagon"))]
fn main() {
    eprintln!("build with --features hexagon");
}
