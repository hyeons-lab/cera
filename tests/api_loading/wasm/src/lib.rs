use wasm_bindgen::prelude::*;

#[derive(Clone)]
#[wasm_bindgen(getter_with_clone)]
pub struct ModelInfo {
    pub requested_context: u32,
    pub capacity: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub backend: String,
    pub draft_model: Option<String>,
    pub gpu_depthformer: bool,
    pub chat_template: Option<String>,
}

#[wasm_bindgen]
pub fn info_for_probe(model: &cera_wasm::GenerativeModel) -> ModelInfo {
    let model = model.core_for_loading_probe();
    let defaults = model.default_generate_opts();
    ModelInfo {
        requested_context: model.config().context_size as u32,
        capacity: model.model().config().max_seq_len as u32,
        temperature: defaults.temperature,
        top_p: defaults.top_p,
        top_k: defaults.top_k,
        min_p: defaults.min_p,
        repetition_penalty: defaults.repetition_penalty,
        backend: format!("{:?}", model.config().backend),
        draft_model: model
            .config()
            .draft_model
            .as_ref()
            .map(|p| p.display().to_string()),
        gpu_depthformer: model.config().gpu_depthformer,
        chat_template: model.manifest().chat_template.clone(),
    }
}

#[wasm_bindgen]
pub fn defaults_for_probe(model: &cera_wasm::GenerativeModel) -> cera_wasm::GenerationDefaults {
    (&model
        .core_for_loading_probe()
        .manifest()
        .generation_defaults)
        .into()
}

#[wasm_bindgen]
pub fn engines_share_for_probe(
    model: &cera_wasm::GenerativeModel,
    engine: &cera_wasm::CeraEngine,
) -> bool {
    engine.shares_core_for_probe(model.core_for_loading_probe().engine())
}

// Synthetic representation control only; no future core model is loaded.
#[wasm_bindgen]
pub struct ProbeFutureHandle;

#[wasm_bindgen]
impl ProbeFutureHandle {
    pub fn kind(&self) -> String {
        "future-probe".into()
    }
    #[wasm_bindgen(js_name = asGenerative)]
    pub fn as_generative(&self) -> Option<cera_wasm::GenerativeModel> {
        None
    }
}

#[wasm_bindgen]
pub fn future_handle_for_probe() -> ProbeFutureHandle {
    ProbeFutureHandle
}

#[wasm_bindgen]
pub fn session_config_for_probe(session: &cera_wasm::Session) -> cera_wasm::SessionConfig {
    session.config_for_loading_probe()
}

#[path = "../../context.rs"]
mod context;

// Exercise the same native diagnostic conversion on wasm32.
#[wasm_bindgen]
pub fn native_context_size_for_probe(context_size: u64) -> Result<u64, JsValue> {
    context::native_context_size(context_size)
        .map(|value| value as u64)
        .map_err(|detail| {
            let error = js_sys::Error::new(&detail);
            for (key, value) in [
                ("code", "InvalidConfig".to_string()),
                ("field", "context_size".into()),
                ("value", context_size.to_string()),
                ("reason", "out_of_range".into()),
            ] {
                js_sys::Reflect::set(&error, &key.into(), &value.into())
                    .expect("new error properties");
            }
            error.into()
        })
}
