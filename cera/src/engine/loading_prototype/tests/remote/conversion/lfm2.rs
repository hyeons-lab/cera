//! The streaming converter writes the llama.cpp LFM2 layout: the same checks as
//! `tests/convert_lfm2_layout.rs` makes of the local converter, through the HTTP path.

use super::super::http::Context;
use super::*;
use crate::bundle::HfSpec;
use crate::convert::{QuantizeOptions, TargetQuant, stream_quantize_hf_repo};

const REPO: &str = "lfm2";
const JINJA: &str = "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}";

/// One shard with the tensors the layout checks look at.
fn shard() -> Vec<u8> {
    let tensors: [(&str, Vec<usize>); 4] = [
        ("model.embed_tokens.weight", vec![8, 32]),
        ("model.embedding_norm.weight", vec![32]),
        ("model.layers.0.operator_norm.weight", vec![32]),
        ("model.layers.0.conv.conv.weight", vec![32, 1, 3]),
    ];
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape) in tensors {
        let start = data.len();
        let count: usize = shape.iter().product();
        for i in 0..count {
            data.extend_from_slice(&(i as f32 * 0.25 + 1.0).to_le_bytes());
        }
        header.insert(
            name.into(),
            json!({"dtype":"F32", "shape":shape, "data_offsets":[start, data.len()]}),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&data);
    bytes
}

fn hf_config(model_type: &str) -> serde_json::Value {
    json!({
        "model_type": model_type, "hidden_size": 32, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "intermediate_size": 96, "block_ff_dim": 96, "block_auto_adjust_ff_dim": true,
        "block_ffn_dim_multiplier": 1.0, "block_multiple_of": 32,
        "norm_eps": 1e-5, "conv_L_cache": 3, "vocab_size": 8,
        "max_position_embeddings": 256,
        "layer_types": ["conv", "full_attention"],
        "rope_parameters": {"rope_theta": 1000000.0},
        "bos_token_id": 1, "eos_token_id": 2,
    })
}

fn routes(model_type: &str) -> HashMap<String, Response> {
    let mut routes = fixture::routes(REPO, "main", &[("model.safetensors".into(), shard())], true);
    let prefix = format!("/fixture/{REPO}/resolve/{}/", fixture::commit("main"));
    let replace = |routes: &mut HashMap<String, Response>, name: &str, body: String| {
        routes.insert(format!("{prefix}{name}"), Response::bytes(body));
    };
    replace(
        &mut routes,
        "config.json",
        hf_config(model_type).to_string(),
    );
    replace(
        &mut routes,
        "tokenizer.json",
        json!({
            "model": {"type": "BPE", "vocab": {"<|pad|>": 0, "<s>": 1, "</s>": 2, "a": 3, "b": 4, "ab": 5},
                      "merges": ["a b"]},
            "added_tokens": [{"id": 0, "content": "<|pad|>", "special": true},
                             {"id": 1, "content": "<s>", "special": true},
                             {"id": 2, "content": "</s>", "special": true}],
            "post_processor": {"type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": "<s>"}}, {"Sequence": {"id": "A"}}]},
        })
        .to_string(),
    );
    // Transformers 5 keeps the template out of tokenizer_config.json
    replace(
        &mut routes,
        "tokenizer_config.json",
        json!({"bos_token": "<s>", "eos_token": "</s>"}).to_string(),
    );
    replace(&mut routes, "chat_template.jinja", JINJA.into());
    routes
}

fn options(cfg: &LoadConfig, progress: Arc<dyn DownloadProgress>) -> QuantizeOptions {
    QuantizeOptions {
        target_quant: TargetQuant::F32,
        cache_dir: cfg.bundle_repo.as_ref().unwrap().store_dir().into(),
        auth_token: None,
        progress: Some(progress),
        ..Default::default()
    }
}

#[test]
fn streaming_conversion_writes_the_llama_cpp_lfm2_layout() {
    isolated_with_ranges(
        "conversion::lfm2::streaming_conversion_writes_the_llama_cpp_lfm2_layout",
        || routes("lfm2"),
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = config(&ctx.root, &progress);
            let spec = HfSpec::parse(&format!("fixture/{REPO}:F32")).unwrap();
            stream_quantize_hf_repo(&spec, options(&cfg, progress)).unwrap();
            let gguf = GgufFile::open(&cache_dir(&cfg, REPO, "F32").join("model.gguf")).unwrap();

            assert_eq!(gguf.architecture(), Some("lfm2"));
            assert_eq!(
                gguf.get_i32_array("lfm2.attention.head_count_kv"),
                Some(vec![0, 1])
            );
            // int(2 * 96 / 3) = 64, already a multiple of 32
            assert_eq!(gguf.get_u32("lfm2.feed_forward_length"), Some(64));
            assert_eq!(gguf.get_u32("lfm2.shortconv.l_cache"), Some(3));
            assert_eq!(gguf.get_f32("lfm2.rope.freq_base"), Some(1_000_000.0));
            assert_eq!(
                gguf.get_f32("lfm2.attention.layer_norm_rms_epsilon"),
                Some(1e-5)
            );

            assert!(gguf.tensors.contains_key("token_embd_norm.weight"));
            assert_eq!(
                gguf.tensors["blk.0.shortconv.conv.weight"].shape,
                vec![3, 32]
            );

            let tokens = gguf.get_string_array("tokenizer.ggml.tokens").unwrap();
            assert_eq!(tokens.len(), 8);
            assert_eq!(tokens[6], "[PAD6]");
            assert_eq!(gguf.get_bool("tokenizer.ggml.add_bos_token"), Some(true));
            // fetched from chat_template.jinja
            assert_eq!(gguf.get_str("tokenizer.chat_template"), Some(JINJA));
        },
        |requests, _ranges| {
            assert_eq!(
                count(
                    requests,
                    "GET",
                    &format!(
                        "/fixture/{REPO}/resolve/{}/chat_template.jinja",
                        fixture::commit("main")
                    )
                ),
                1
            );
        },
    );
}

#[test]
fn streaming_conversion_refuses_lfm2_moe_before_fetching_weights() {
    isolated_with_ranges(
        "conversion::lfm2::streaming_conversion_refuses_lfm2_moe_before_fetching_weights",
        || routes("lfm2_moe"),
        |ctx| {
            let progress = Arc::new(Progress::default());
            let cfg = config(&ctx.root, &progress);
            let spec = HfSpec::parse(&format!("fixture/{REPO}:F32")).unwrap();
            let err = stream_quantize_hf_repo(&spec, options(&cfg, progress))
                .expect_err("MoE conversion is not implemented");
            assert!(err.to_string().contains("LFM2-MoE"), "{err}");
            assert_no_conversion_artifacts(&ctx);
        },
        |_requests, ranges| {
            assert!(
                ranges.is_empty(),
                "no shard bytes may be fetched: {ranges:?}"
            );
        },
    );
}

/// The LFM2 fixture with one file's response replaced.
fn file_route(name: &'static str, response: Response) -> impl Fn() -> HashMap<String, Response> {
    move || {
        let mut routes = routes("lfm2");
        routes.insert(url(name), response.clone());
        routes
    }
}

fn url(name: &str) -> String {
    format!("/fixture/{REPO}/resolve/{}/{name}", fixture::commit("main"))
}

/// Convert the fixture to F32 and open the result.
fn convert(ctx: &Context) -> Result<GgufFile, crate::CeraError> {
    let progress = Arc::new(Progress::default());
    let cfg = config(&ctx.root, &progress);
    let spec = HfSpec::parse(&format!("fixture/{REPO}:F32")).unwrap();
    stream_quantize_hf_repo(&spec, options(&cfg, progress))?;
    Ok(GgufFile::open(&cache_dir(&cfg, REPO, "F32").join("model.gguf")).unwrap())
}

/// A refused conversion leaves nothing a later call could mistake for a result: no GGUF,
/// no temp file, no checkpoint and no receipt.
fn assert_no_conversion_artifacts(ctx: &Context) {
    let progress = Arc::new(Progress::default());
    let cfg = config(&ctx.root, &progress);
    let left: Vec<_> = fs::read_dir(cache_dir(&cfg, REPO, "F32"))
        .map(|dir| dir.flatten().map(|e| e.file_name()).collect())
        .unwrap_or_default();
    assert!(left.is_empty(), "a refused conversion left {left:?}");
}

#[test]
fn a_repo_without_chat_template_jinja_converts_without_a_template() {
    isolated_with_ranges(
        "conversion::lfm2::a_repo_without_chat_template_jinja_converts_without_a_template",
        file_route("chat_template.jinja", Response::status(404)),
        |ctx| {
            let gguf = convert(&ctx).unwrap();
            assert_eq!(gguf.get_str("tokenizer.chat_template"), None);
        },
        |_requests, _ranges| {},
    );
}

#[test]
fn a_chat_template_that_cannot_be_fetched_fails_the_conversion() {
    // 403 is not retried. Treating it like a 404 would convert to a GGUF with no
    // chat template and cache it as complete.
    isolated_with_ranges(
        "conversion::lfm2::a_chat_template_that_cannot_be_fetched_fails_the_conversion",
        file_route("chat_template.jinja", Response::status(403)),
        |ctx| {
            let err = convert(&ctx)
                .err()
                .expect("an unreadable template must not be skipped");
            assert!(err.to_string().contains("chat_template.jinja"), "{err}");
            assert_no_conversion_artifacts(&ctx);
        },
        |requests, _ranges| {
            assert_eq!(count(requests, "GET", &url("chat_template.jinja")), 1);
        },
    );
}

#[test]
fn a_chat_template_that_is_not_text_fails_the_conversion() {
    isolated_with_ranges(
        "conversion::lfm2::a_chat_template_that_is_not_text_fails_the_conversion",
        file_route("chat_template.jinja", Response::bytes([0xff, 0xfe])),
        |ctx| {
            let err = convert(&ctx)
                .err()
                .expect("a template that is not UTF-8 must not be skipped");
            assert!(err.to_string().contains("chat_template.jinja"), "{err}");
            assert_no_conversion_artifacts(&ctx);
        },
        |_requests, _ranges| {},
    );
}

/// `tokenizer_config.json` carries the special tokens, the `add_*_token` overrides and
/// often the template, so one that is present but unusable fails the conversion.
fn unusable_tokenizer_config(name: &str, response: Response) {
    isolated_with_ranges(
        &format!("conversion::lfm2::{name}"),
        file_route("tokenizer_config.json", response),
        |ctx| {
            let err = convert(&ctx)
                .err()
                .expect("an unusable tokenizer_config must not be skipped");
            assert!(err.to_string().contains("tokenizer_config.json"), "{err}");
            assert_no_conversion_artifacts(&ctx);
        },
        |_requests, _ranges| {},
    );
}

#[test]
fn a_tokenizer_config_that_cannot_be_fetched_fails_the_conversion() {
    unusable_tokenizer_config(
        "a_tokenizer_config_that_cannot_be_fetched_fails_the_conversion",
        Response::status(403),
    );
}

#[test]
fn a_tokenizer_config_that_is_not_json_fails_the_conversion() {
    unusable_tokenizer_config(
        "a_tokenizer_config_that_is_not_json_fails_the_conversion",
        Response::bytes("{ nope"),
    );
}

#[test]
fn chat_template_jinja_is_not_fetched_when_tokenizer_config_has_the_template() {
    let config = json!({"bos_token": "<s>", "eos_token": "</s>", "chat_template": "{{ x }}"});
    isolated_with_ranges(
        "conversion::lfm2::chat_template_jinja_is_not_fetched_when_tokenizer_config_has_the_template",
        // the standalone file would fail the conversion if it were consulted
        move || {
            let mut routes = file_route("chat_template.jinja", Response::status(403))();
            routes.insert(
                url("tokenizer_config.json"),
                Response::bytes(config.to_string()),
            );
            routes
        },
        |ctx| {
            let gguf = convert(&ctx).unwrap();
            assert_eq!(gguf.get_str("tokenizer.chat_template"), Some("{{ x }}"));
        },
        |requests, _ranges| {
            assert_eq!(count(requests, "GET", &url("chat_template.jinja")), 0);
        },
    );
}

#[test]
fn a_token_id_past_vocab_size_is_refused_before_fetching_weights() {
    isolated_with_ranges(
        "conversion::lfm2::a_token_id_past_vocab_size_is_refused_before_fetching_weights",
        || {
            let mut routes = routes("lfm2");
            // `vocab_size` is 8, so id 8 is one past the embedding
            routes.insert(
                url("tokenizer.json"),
                Response::bytes(
                    json!({
                        "model": {"type": "BPE", "vocab": {"a": 0, "b": 8}, "merges": []},
                    })
                    .to_string(),
                ),
            );
            routes
        },
        |ctx| {
            let err = convert(&ctx)
                .err()
                .expect("a token id past vocab_size must not be converted");
            assert!(err.to_string().contains("vocab_size"), "{err}");
            assert_no_conversion_artifacts(&ctx);
        },
        |_requests, ranges| {
            assert!(
                ranges.is_empty(),
                "no shard bytes may be fetched: {ranges:?}"
            );
        },
    );
}
