use super::*;
use crate::model::{ModelConfig, ModelSessionGate, ScalarMultipliers};
use std::sync::atomic::{AtomicU8, AtomicUsize};

struct SharedModel(ModelConfig);

impl Model for SharedModel {
    fn config(&self) -> &ModelConfig {
        &self.0
    }
    fn forward(&self, tokens: &[u32], _: usize, state: &mut InferenceState) -> Vec<f32> {
        state.seq_len += tokens.len();
        vec![0.0, 1.0]
    }
}

fn config() -> ModelConfig {
    ModelConfig {
        architecture: "ownership-test".into(),
        n_layers: 0,
        hidden_size: 2,
        intermediate_size: 2,
        n_heads: 1,
        n_kv_heads: 1,
        head_dim: 2,
        vocab_size: 2,
        max_seq_len: 16,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        block_types: Vec::new(),
        conv_kernel_size: None,
        ssm: None,
        kv_heads_per_layer: Vec::new(),
        scalars: ScalarMultipliers::default(),
        moe: None,
        is_causal: true,
        class_labels: Vec::new(),
    }
}

struct ReservedModel {
    inner: SharedModel,
    gate: ModelSessionGate,
    configured: AtomicUsize,
    failure: AtomicU8,
    /// Whether the model hands out embedding rows for image markers (the CPU, wgpu and Hexagon models
    /// do; native Metal does not).
    marker_rows: std::sync::atomic::AtomicBool,
}

impl ReservedModel {
    fn new() -> Self {
        Self {
            inner: SharedModel(config()),
            gate: ModelSessionGate::default(),
            configured: AtomicUsize::new(0),
            failure: AtomicU8::new(0),
            marker_rows: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

impl Model for ReservedModel {
    fn config(&self) -> &ModelConfig {
        self.inner.config()
    }
    fn embed_image_marker_rows(&self, tokens: &[u32]) -> Option<Vec<f32>> {
        self.marker_rows
            .load(Ordering::Relaxed)
            .then(|| vec![0.0; tokens.len() * self.inner.config().hidden_size])
    }
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        self.inner.forward(tokens, pos, state)
    }
    fn acquire_session(&self) -> Result<Option<ModelSessionLease>, CeraError> {
        self.gate.try_acquire().map(Some)
    }
    fn configure_kv_compression(&self, _: &KvCompression) -> Result<(), CeraError> {
        self.configured.fetch_add(1, Ordering::Relaxed);
        match self.failure.load(Ordering::Relaxed) {
            1 => Err(CeraError::Backend("configuration rejected".into())),
            2 => panic!("configuration panic"),
            _ => Ok(()),
        }
    }
}

fn create(model: Arc<dyn Model>) -> Result<Session, CeraError> {
    Session::new(
        model,
        Arc::new(BpeTokenizer::empty_for_test()),
        ModalityCapabilities::text_only(),
        SessionConfig::default(),
    )
}

#[test]
fn busy_precedes_configuration_and_reset_retains_ownership() {
    let model = Arc::new(ReservedModel::new());
    let mut active = create(model.clone()).unwrap();
    active.append_tokens(&[0, 1]).unwrap();
    model.failure.store(1, Ordering::Relaxed);
    assert!(matches!(create(model.clone()), Err(CeraError::Busy)));
    assert_eq!(model.configured.load(Ordering::Relaxed), 1);
    assert_eq!(active.position(), 2);
    assert!(active.reset().is_err());
    assert!(matches!(create(model.clone()), Err(CeraError::Busy)));
    model.failure.store(0, Ordering::Relaxed);
    active.reset().unwrap();
    active.cancel();
    assert!(matches!(create(model.clone()), Err(CeraError::Busy)));
    active.clear_cancel();
    active.append_tokens(&[1]).unwrap();
    let observed_position = active.position_handle();
    let cancel = active.cancel_handle();
    drop(active);
    let successor = create(model).unwrap();
    assert_eq!(successor.position(), 0);
    // Observation/cancellation handles do not own the inference context.
    assert_eq!(observed_position.load(Ordering::Relaxed), 1);
    assert!(!cancel.load(Ordering::Relaxed));
}

#[test]
fn failed_and_panicking_construction_release_ownership() {
    for failure in [1, 2] {
        let model = Arc::new(ReservedModel::new());
        model.failure.store(failure, Ordering::Relaxed);
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| create(model.clone())));
        if failure == 1 {
            assert!(matches!(outcome, Ok(Err(CeraError::Backend(_)))));
        } else {
            assert!(outcome.is_err());
        }
        model.failure.store(0, Ordering::Relaxed);
        assert!(create(model).is_ok());
    }
    let mut invalid = ReservedModel::new();
    invalid.inner.0.n_heads = usize::MAX;
    let model = Arc::new(invalid);
    assert!(matches!(
        create(model.clone()),
        Err(CeraError::OutOfMemory { .. })
    ));
    assert!(model.gate.try_acquire().is_ok());
}

#[test]
fn default_model_hook_preserves_multiple_live_sessions() {
    let model = Arc::new(SharedModel(config()));
    let mut a = create(model.clone()).unwrap();
    let mut b = create(model).unwrap();
    a.append_tokens(&[0, 1]).unwrap();
    b.append_tokens(&[1]).unwrap();
    a.reset().unwrap();
    assert_eq!(b.position(), 1);
    b.append_tokens(&[0]).unwrap();
    assert_eq!(b.position(), 2);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn racing_session_constructors_have_one_owner_and_release_across_threads() {
    let model = Arc::new(ReservedModel::new());
    let start = Arc::new(std::sync::Barrier::new(8));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let model = model.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                start.wait();
                create(model)
            })
        })
        .collect();
    // Returned Sessions stay alive in their JoinHandles until collected here.
    let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|r| matches!(r, Err(CeraError::Busy)))
            .count(),
        7
    );
    assert_eq!(model.configured.load(Ordering::Relaxed), 1);
    drop(results);
    assert!(create(model).is_ok());
}

struct DropProbe {
    model: Arc<ReservedModel>,
    saw_busy: Arc<AtomicBool>,
}

impl crate::spec::Drafter for DropProbe {
    fn clone_drafter(&self) -> Box<dyn crate::spec::Drafter> {
        unreachable!()
    }
    fn reset(&mut self) {}
    fn draft(&mut self, _: &[u32], _: usize) -> Vec<u32> {
        Vec::new()
    }
}

impl Drop for DropProbe {
    fn drop(&mut self) {
        self.saw_busy.store(
            matches!(self.model.gate.try_acquire(), Err(CeraError::Busy)),
            Ordering::Relaxed,
        );
    }
}

#[test]
fn session_drops_owned_resources_before_releasing_the_lease() {
    let model = Arc::new(ReservedModel::new());
    let mut active = create(model.clone()).unwrap();
    let saw_busy = Arc::new(AtomicBool::new(false));
    active.drafter = Some(Box::new(DropProbe {
        model: model.clone(),
        saw_busy: saw_busy.clone(),
    }));
    drop(active);
    assert!(saw_busy.load(Ordering::Relaxed));
    assert!(create(model).is_ok());
}

/// Whether more rows fit is decided before they are produced: a session at the context limit refuses
/// them without touching its state, which is what lets an oversize tiled image fail before the vision
/// tower runs, and the decision is the one `append_embeddings` makes.
#[test]
fn ensure_rows_fit_refuses_overflow_without_mutating() {
    let mut session = create(Arc::new(ReservedModel::new())).unwrap();
    let max = 16usize;
    assert!(
        session.ensure_rows_fit(max).is_ok(),
        "exactly the context fits"
    );
    match session.ensure_rows_fit(max + 1) {
        Err(CeraError::ContextOverflow { max_seq_len, by }) => {
            assert_eq!((max_seq_len as usize, by), (max, 1));
        }
        other => panic!("expected ContextOverflow, got {other:?}"),
    }
    session.append_tokens(&[0, 1]).unwrap();
    assert!(session.ensure_rows_fit(max - 2).is_ok());
    assert!(matches!(
        session.ensure_rows_fit(max - 1),
        Err(CeraError::ContextOverflow { by: 1, .. })
    ));
    assert!(matches!(
        session.ensure_rows_fit(usize::MAX),
        Err(CeraError::Backend(_))
    ));
    assert_eq!(session.position(), 2, "checking must not move the session");
}

/// A preprocessed image whose patch grid is `gw x gh`, enough for the row arithmetic.
#[cfg(feature = "vl-preprocess")]
fn grid_only(gw: usize, gh: usize) -> crate::model::vision_preprocessor::PreprocessedImage {
    crate::model::vision_preprocessor::PreprocessedImage {
        pixels: Vec::new(),
        target_w: gw * 16,
        target_h: gh * 16,
        grid_w: gw,
        grid_h: gh,
    }
}

/// Tiling is offered only when the tokenizer resolves the marker tokens (each to exactly one id) and
/// the model hands out their embedding rows: a missing half turns it off, so a large image is
/// preprocessed as one image instead of as a tile set that would be thrown away.
#[cfg(feature = "vl-preprocess")]
#[test]
fn tiling_needs_both_marker_tokens_and_marker_rows() {
    let specials = [("<|img_row_1_col_1|>", 10), ("<|img_thumbnail|>", 11)];
    let session_with = |model: &Arc<ReservedModel>, tokenizer: BpeTokenizer| {
        Session::new(
            model.clone(),
            Arc::new(tokenizer),
            ModalityCapabilities::text_only(),
            SessionConfig::default(),
        )
        .unwrap()
    };
    let model = Arc::new(ReservedModel::new());
    model.marker_rows.store(true, Ordering::Relaxed);
    let vocab = || BpeTokenizer::empty_for_test().with_special_texts_for_testing(&specials);

    // Both halves present: the shape `tile_marker_rows` hands the splice (one row per tile, one thumbnail).
    let session = session_with(&model, vocab());
    assert!(session.tiling_available());
    let (tiles, thumb) = session.tile_marker_rows(1, 1).unwrap();
    assert_eq!((tiles.len(), tiles[0].len(), thumb.len()), (1, 2, 2));
    drop(session);

    // The model cannot hand out rows (native Metal): off.
    model.marker_rows.store(false, Ordering::Relaxed);
    let session = session_with(&model, vocab());
    assert!(!session.tiling_available());
    drop(session);

    // The vocabulary lacks the markers: off.
    model.marker_rows.store(true, Ordering::Relaxed);
    let session = session_with(&model, BpeTokenizer::empty_for_test());
    assert!(!session.tiling_available());
}

/// An image that cannot fit the context is refused before any vision-tower pass, a layout is budgeted
/// together with the rows reserved for the rest of the same prefill, and a model that cannot embed the
/// tile markers encodes only the thumbnail (one pass, not one per tile). The tower and the marker
/// lookup are stand-ins (the fixture's tokenizer has no vocabulary), so the order of check and pass is
/// what is under test. The session's context is 16 rows; a grid of 8x8 patches at merge factor 2 is
/// 16 rows.
#[cfg(feature = "vl-preprocess")]
#[test]
fn layouts_are_budgeted_before_any_tower_pass() {
    use crate::model::vision_preprocessor::{PreprocessedImage, PreprocessedLayout, TiledImage};
    let session = create(Arc::new(ReservedModel::new())).unwrap();
    // No vocabulary and no marker rows: the session must preprocess one image, not a tile set.
    assert!(!session.tiling_available());
    let calls = std::cell::Cell::new(0usize);
    let tower = |p: &PreprocessedImage| {
        calls.set(calls.get() + 1);
        let n = (p.grid_w / 2) * (p.grid_h / 2);
        Ok((vec![0.0f32; n * 2], n))
    };
    let single = |gw, gh| PreprocessedLayout::Single(grid_only(gw, gh));
    let no_markers = |_: usize, _: usize| Err("no marker rows".to_string());
    // One hidden_size (2) row per tile marker and one for the thumbnail marker.
    let markers = |c: usize, r: usize| Ok((vec![vec![0.0f32; 2]; c * r], vec![0.0f32; 2]));

    // Exactly the context fits and runs one pass.
    let (_, n) = session
        .encode_layout_rows_with(&single(8, 8), 2, 0, no_markers, &tower)
        .unwrap();
    assert_eq!((n, calls.get()), (16, 1));

    // One row over is refused with no further pass.
    calls.set(0);
    assert!(matches!(
        session.encode_layout_rows_with(&single(10, 8), 2, 0, no_markers, &tower),
        Err(CeraError::ContextOverflow { .. })
    ));
    assert_eq!(calls.get(), 0, "an oversize image must not reach the tower");

    // Rows already committed to the same prefill count against the image.
    assert!(matches!(
        session.encode_layout_rows_with(&single(8, 8), 2, 1, no_markers, &tower),
        Err(CeraError::ContextOverflow { .. })
    ));
    assert_eq!(
        calls.get(),
        0,
        "reserved rows must be budgeted before the pass"
    );

    // No marker rows (this model cannot hand them out): the thumbnail alone is encoded, once.
    let tiled = |thumb: PreprocessedImage| {
        PreprocessedLayout::Tiled(TiledImage {
            cols: 2,
            rows: 1,
            tiles: vec![grid_only(8, 8), grid_only(8, 8)],
            thumbnail: thumb,
        })
    };
    let (_, n) = session
        .encode_layout_rows_with(&tiled(grid_only(4, 4)), 2, 0, no_markers, &tower)
        .unwrap();
    assert_eq!((n, calls.get()), (4, 1), "thumbnail only, no tile passes");
    calls.set(0);
    assert!(matches!(
        session.encode_layout_rows_with(&tiled(grid_only(10, 8)), 2, 0, no_markers, &tower),
        Err(CeraError::ContextOverflow { .. })
    ));
    assert_eq!(
        calls.get(),
        0,
        "an oversize thumbnail must not reach the tower"
    );

    // Markers available: the whole layout (tiles, thumbnail and their marker rows) is budgeted first.
    // Two 16-row tiles and a 4-row thumbnail with 3 marker rows is 39 rows: refused with no pass.
    let big_tiles = PreprocessedLayout::Tiled(TiledImage {
        cols: 2,
        rows: 1,
        tiles: vec![grid_only(8, 8), grid_only(8, 8)],
        thumbnail: grid_only(4, 4),
    });
    calls.set(0);
    assert!(matches!(
        session.encode_layout_rows_with(&big_tiles, 2, 0, markers, &tower),
        Err(CeraError::ContextOverflow { .. })
    ));
    assert_eq!(
        calls.get(),
        0,
        "an oversize tile set must not reach the tower"
    );
    // Two 4-row tiles, a 4-row thumbnail and 3 marker rows are 15 rows: three passes, spliced.
    let small_tiles = || {
        PreprocessedLayout::Tiled(TiledImage {
            cols: 2,
            rows: 1,
            tiles: vec![grid_only(4, 4), grid_only(4, 4)],
            thumbnail: grid_only(4, 4),
        })
    };
    let (_, n) = session
        .encode_layout_rows_with(&small_tiles(), 2, 0, markers, &tower)
        .unwrap();
    assert_eq!((n, calls.get()), (15, 3));
    // Reserved rows that leave too little room: refused before the first pass.
    calls.set(0);
    assert!(matches!(
        session.encode_layout_rows_with(&small_tiles(), 2, 2, markers, &tower),
        Err(CeraError::ContextOverflow { .. })
    ));
    assert_eq!(
        calls.get(),
        0,
        "reserved rows must be budgeted for a tile set too"
    );
}

/// Records the KV mode the session resolved and handed to the model.
struct ModeProbe {
    inner: SharedModel,
    f16: bool,
    seen: std::sync::Mutex<Vec<KvCompression>>,
}

impl Model for ModeProbe {
    fn config(&self) -> &ModelConfig {
        self.inner.config()
    }
    fn forward(&self, tokens: &[u32], pos: usize, state: &mut InferenceState) -> Vec<f32> {
        self.inner.forward(tokens, pos, state)
    }
    fn f16_kv_supported(&self) -> bool {
        self.f16
    }
    fn configure_kv_compression(&self, c: &KvCompression) -> Result<(), CeraError> {
        self.seen.lock().unwrap().push(c.clone());
        Ok(())
    }
}

/// The library default is f16, but only for a model that honors it: any other
/// model must be configured with its own uncompressed KV, and an explicit mode
/// is passed through untouched.
#[test]
fn default_f16_applies_only_to_models_that_honor_it() {
    let session = |f16: bool, kv: Option<KvCompression>| {
        let model = Arc::new(ModeProbe {
            inner: SharedModel(config()),
            f16,
            seen: Default::default(),
        });
        let mut cfg = SessionConfig::default();
        if let Some(kv) = kv {
            cfg.kv_compression = kv;
        }
        Session::new(
            model.clone(),
            Arc::new(BpeTokenizer::empty_for_test()),
            ModalityCapabilities::text_only(),
            cfg,
        )
        .unwrap();
        let seen = model.seen.lock().unwrap();
        seen.iter()
            .map(KvCompression::cache_tag)
            .collect::<Vec<_>>()
    };
    let none = KvCompression::None.cache_tag();
    let f16 = KvCompression::F16.cache_tag();
    let tq = KvCompression::turboquant(7).cache_tag();
    let want_default = if cfg!(target_arch = "wasm32") {
        &none
    } else {
        &f16
    };
    assert_eq!(session(true, None), std::slice::from_ref(want_default));
    assert_eq!(session(false, None), std::slice::from_ref(&none));
    // An explicit full-precision request is never upgraded.
    assert_eq!(session(true, Some(KvCompression::None)), [none]);
    // Nor is another mode rewritten.
    assert_eq!(session(false, Some(KvCompression::turboquant(7))), [tq]);
}
