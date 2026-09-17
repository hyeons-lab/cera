use std::sync::{Arc, Mutex};

use super::core_api::gguf::{GgufFile, GgufValue};
use super::core_api::kv_cache::InferenceState;
use super::core_api::model::{BlockType, Model, ModelConfig, ScalarMultipliers};
use super::core_api::session::{
    FinishReason, ModalityCapabilities, ModalitySink, Session, SessionConfig,
};
use super::core_api::tokenizer::BpeTokenizer;

use super::contract::{LFM2_5_TEMPLATE, TEMPLATE};

fn make_fixture_tokenizer(template: &str) -> Arc<BpeTokenizer> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.resize(32, 0);
    let mut gguf = GgufFile::from_bytes(bytes.into()).unwrap();
    let mut vocab = vec![
        "<pad>".to_owned(),
        "<|startoftext|>".into(),
        "<|image_start|>".into(),
        "<|image_end|>".into(),
        "<image>".into(),
        "<|reserved_4|>".into(),
        "<|im_start|>".into(),
        "<|im_end|>".into(),
        "Ġ".into(),
        "Ċ".into(),
    ];
    vocab.extend((33u8..=126).map(|b| (b as char).to_string()));
    let types = (0..vocab.len())
        .map(|i| GgufValue::I32(if i < 8 { 3 } else { 1 }))
        .collect();
    gguf.metadata.insert(
        "tokenizer.ggml.tokens".into(),
        GgufValue::Array(vocab.into_iter().map(GgufValue::String).collect()),
    );
    gguf.metadata
        .insert("tokenizer.ggml.token_type".into(), GgufValue::Array(types));
    gguf.metadata
        .insert("tokenizer.ggml.bos_token_id".into(), GgufValue::U32(1));
    gguf.metadata
        .insert("tokenizer.ggml.eos_token_id".into(), GgufValue::U32(7));
    gguf.metadata.insert(
        "tokenizer.chat_template".into(),
        GgufValue::String(template.into()),
    );
    Arc::new(BpeTokenizer::from_gguf(&gguf).unwrap())
}

/// Small ASCII vocabulary for offline contract tests. Public-tokenizer tests
/// separately load the exact full GGUF, including all vocabulary and merges.
pub fn tokenizer() -> Arc<BpeTokenizer> {
    make_fixture_tokenizer(TEMPLATE)
}

/// Small ASCII vocabulary configured with the canonical LFM2.5 chat template.
pub fn lfm2_5_tokenizer() -> Arc<BpeTokenizer> {
    make_fixture_tokenizer(LFM2_5_TEMPLATE)
}

#[derive(Default)]
pub struct Sink {
    pub tokens: Vec<u32>,
    pub done: Vec<FinishReason>,
}

impl ModalitySink for Sink {
    fn on_text_tokens(&mut self, tokens: &[u32]) {
        self.tokens.extend_from_slice(tokens);
    }
    fn on_done(&mut self, reason: FinishReason) {
        self.done.push(reason);
    }
}

/// One attention layer with two-wide heads: the smallest shape whose cache rows
/// can carry a token id and its position for later inspection.
pub fn model_config(architecture: &str, vocab_size: usize) -> ModelConfig {
    ModelConfig {
        architecture: architecture.into(),
        n_layers: 1,
        hidden_size: 2,
        intermediate_size: 2,
        n_heads: 1,
        n_kv_heads: 1,
        head_dim: 2,
        vocab_size,
        max_seq_len: 8192,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        block_types: vec![BlockType::Attention],
        conv_kernel_size: None,
        ssm: None,
        kv_heads_per_layer: vec![1],
        scalars: ScalarMultipliers::default(),
        moe: None,
        is_causal: true,
        class_labels: Vec::new(),
    }
}

/// Scripted next token: `answer` after anything else, then EOS after `answer`.
pub fn scripted_logits(tokens: &[u32], answer: u32, vocab_size: usize) -> Vec<f32> {
    let next = if tokens.last() == Some(&answer) {
        7
    } else {
        answer
    };
    let mut logits = vec![f32::NEG_INFINITY; vocab_size];
    logits[next as usize] = 0.0;
    logits
}

/// Records inputs consumed by the real Session decode loop. Its next-token
/// choice is scripted; this tests control flow, not numerical model quality.
pub struct TraceModel {
    config: ModelConfig,
    pub resident: Mutex<Vec<u32>>,
    output: u32,
}

impl TraceModel {
    pub fn session(tokenizer: Arc<BpeTokenizer>) -> (Arc<Self>, Session) {
        let output = tokenizer.encode("a")[0];
        let model = Arc::new(Self {
            config: model_config("chat-boundary-test", tokenizer.vocab_size()),
            resident: Mutex::new(Vec::new()),
            output,
        });
        let session = Session::new(
            model.clone(),
            tokenizer,
            ModalityCapabilities::text_only(),
            SessionConfig {
                seed: Some(42),
                ..Default::default()
            },
        )
        .unwrap();
        (model, session)
    }
}

impl Model for TraceModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        let mut resident = self.resident.lock().unwrap();
        assert_eq!(resident.len(), pos);
        resident.extend_from_slice(tokens);
        state.seq_len = resident.len();
        scripted_logits(tokens, self.output, self.config.vocab_size)
    }
}
