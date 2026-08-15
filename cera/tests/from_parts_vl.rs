//! End-to-end cover for [`cera::CeraEngine::from_parts`], the constructor
//! that makes multimodal bundles reachable without a filesystem.
//!
//! Everything here reads the GGUFs into `Vec<u8>` first and hands the engine
//! only bytes, which is the shape a browser is limited to. A path-based test
//! would pass while the in-memory path stayed broken, since the two reach
//! `from_gguf` with different aux-loading rules: the eager mmproj load is
//! gated on `path.is_some()`, so the bytes route depends entirely on the
//! pre-parsed `AuxWeights` this constructor threads through.
//!
//! Requires a local LFM2-VL bundle. Run with:
//!
//! ```sh
//! cargo test -p cera --test from_parts_vl -- --nocapture
//! ```
//!
//! The image-input assertions additionally need the default `vl-preprocess`
//! feature; without it `append_image` is a stub that returns
//! `UnsupportedModality`, so they are compiled out rather than left to fail
//! confusingly.

use std::path::PathBuf;

use cera::manifest::InferenceType;
use cera::{CeraEngine, EngineConfig, ModelBytes};

/// Locate a VL bundle in `~/.leap/models`, returning `(model, mmproj)`.
///
/// Returns `None` (with a printed reason) when the bundle is absent so a
/// developer without the fixture is not blocked. That does mean an absent
/// model reads as a pass, so `vl_fixture_is_present` below fails loudly when
/// `CERA_REQUIRE_VL_FIXTURE` is set: CI can opt in and get a real signal
/// instead of a silent skip.
fn find_vl_bundle() -> Option<(PathBuf, PathBuf)> {
    let root = PathBuf::from(std::env::var("HOME").ok()?).join(".leap/models");
    // Ordered by preference: the smallest bundle that exercises the same code
    // path keeps the test quick.
    let candidates = [
        ("LFM2-VL-450M-Q8_0", "mmproj-LFM2-VL-450M-Q8_0.gguf"),
        ("LFM2-VL-450M-Q4_0", "mmproj-LFM2-VL-450M-Q8_0.gguf"),
        ("LFM2.5-VL-450M-Q4_0", "mmproj-LFM2.5-VL-450M-Q8_0.gguf"),
    ];
    for (dir, mmproj) in candidates {
        let model = root.join(dir).join(format!("{dir}.gguf"));
        let proj = root.join(dir).join(mmproj);
        if model.exists() && proj.exists() {
            return Some((model, proj));
        }
    }
    eprintln!("no LFM2-VL bundle under {}, skipping", root.display());
    None
}

fn cfg() -> EngineConfig {
    // The spread is deliberate, as in `cera-wasm`'s constructors: without the
    // `remote` feature `EngineConfig` collapses to exactly the two fields
    // named here and clippy calls it needless, but a future field would
    // otherwise compile-break this helper.
    #[allow(clippy::needless_update)]
    EngineConfig {
        context_size: 1024,
        backend: cera::BackendPreference::Cpu,
        ..EngineConfig::default()
    }
}

/// Encode a small solid-colour PNG in memory.
///
/// Generated rather than embedded as a byte literal: the first version of
/// this test hand-wrote the PNG and shipped a bad IDAT CRC, which the decoder
/// rejected and which looked exactly like a bug in the image path.
///
/// 64x64 rather than 2x2 because the preprocessor aligns to patch blocks and
/// a sub-patch image is a degenerate case; this asserts the plumbing, not the
/// edge handling.
fn solid_png() -> Vec<u8> {
    let img = image::RgbImage::from_pixel(64, 64, image::Rgb([220u8, 30, 30]));
    let mut out = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut out, image::ImageFormat::Png)
        .expect("encode png");
    out.into_inner()
}

/// The headline: bytes in, a VL-capable engine out. Before `from_parts` there
/// was no way to reach this state without a filesystem.
#[test]
fn from_parts_loads_a_vl_bundle_from_bytes_alone() {
    let Some((model_path, mmproj_path)) = find_vl_bundle() else {
        return;
    };
    let model = std::fs::read(&model_path).expect("read model");
    let mmproj = std::fs::read(&mmproj_path).expect("read mmproj");

    let engine = CeraEngine::from_parts(
        ModelBytes {
            model: model.into(),
            multimodal_projector: Some(mmproj.into()),
            audio_decoder: None,
            inference_type: None,
            chat_template: None,
        },
        cfg(),
    )
    .expect("from_parts should load a VL bundle from bytes");

    let caps = engine.capabilities();
    assert!(
        caps.image_in,
        "an mmproj-carrying bundle must report imageIn; got {caps:?}. \
         If this fails with text_in only, the arch-detect upgrade in \
         resolve_parts_inference_type did not fire."
    );
    assert!(caps.text_out);
}

/// The negative control, and the reason the upgrade rule exists at all: the
/// *same* model GGUF without its mmproj is text-only. If this reported
/// `image_in` the flag would be meaningless.
#[test]
fn the_same_model_without_its_mmproj_is_text_only() {
    let Some((model_path, _)) = find_vl_bundle() else {
        return;
    };
    let model = std::fs::read(&model_path).expect("read model");

    let engine = CeraEngine::from_parts(ModelBytes::text(model), cfg())
        .expect("from_parts should load the LLM half alone");

    assert!(
        !engine.capabilities().image_in,
        "a bare VL model GGUF carries no vision tower and must not claim imageIn"
    );
}

/// An explicit `inference_type` overrides the mmproj upgrade. This is the
/// documented opt-out, so it is worth pinning against a real bundle rather
/// than only in the unit test for the pure resolver.
#[test]
fn explicit_text_type_suppresses_the_upgrade() {
    let Some((model_path, mmproj_path)) = find_vl_bundle() else {
        return;
    };
    let model = std::fs::read(&model_path).expect("read model");
    let mmproj = std::fs::read(&mmproj_path).expect("read mmproj");

    let engine = CeraEngine::from_parts(
        ModelBytes {
            model: model.into(),
            multimodal_projector: Some(mmproj.into()),
            audio_decoder: None,
            inference_type: Some(InferenceType::LlamaCppTextToText),
            chat_template: None,
        },
        cfg(),
    )
    .expect("explicit text type should still load");

    assert!(
        !engine.capabilities().image_in,
        "an explicit text inference_type must win over the mmproj upgrade"
    );
}

/// Corrupt mmproj bytes must not fail the whole load: a broken sidecar
/// degrades to text rather than taking the page down with it.
#[test]
fn a_corrupt_mmproj_degrades_to_text_instead_of_failing() {
    let Some((model_path, _)) = find_vl_bundle() else {
        return;
    };
    let model = std::fs::read(&model_path).expect("read model");

    let engine = CeraEngine::from_parts(
        ModelBytes {
            model: model.into(),
            multimodal_projector: Some(vec![0xDE, 0xAD, 0xBE, 0xEF].into()),
            audio_decoder: None,
            inference_type: None,
            chat_template: None,
        },
        cfg(),
    )
    .expect("a corrupt mmproj must warn, not fail the load");

    assert!(
        !engine.capabilities().image_in,
        "a bundle whose mmproj failed to parse must report imageIn = false, \
         so the flag describes what the engine can do rather than what the \
         caller intended"
    );
}

/// The payoff: an image actually goes through preprocess, the vision encoder,
/// and into the KV cache. `position()` advancing by more than the text tokens
/// alone is the observable signal that image embeddings were spliced in.
#[cfg(feature = "vl-preprocess")]
#[test]
fn append_image_advances_the_kv_cache() {
    let Some((model_path, mmproj_path)) = find_vl_bundle() else {
        return;
    };
    let model = std::fs::read(&model_path).expect("read model");
    let mmproj = std::fs::read(&mmproj_path).expect("read mmproj");

    let engine = CeraEngine::from_parts(
        ModelBytes {
            model: model.into(),
            multimodal_projector: Some(mmproj.into()),
            audio_decoder: None,
            inference_type: None,
            chat_template: None,
        },
        cfg(),
    )
    .expect("from_parts");

    let mut session = engine
        .new_session(cera::SessionConfig::default())
        .expect("new_session");

    assert_eq!(session.position(), 0);
    session
        .append_image(&solid_png())
        .expect("append_image should succeed on a VL bundle loaded from bytes");
    assert!(
        session.position() > 0,
        "append_image must add image tokens to the KV cache"
    );
}

/// A text-only engine rejects images by modality rather than by panicking or
/// silently no-oping.
#[cfg(feature = "vl-preprocess")]
#[test]
fn append_image_on_a_text_bundle_reports_unsupported_modality() {
    let Some((model_path, _)) = find_vl_bundle() else {
        return;
    };
    let model = std::fs::read(&model_path).expect("read model");

    let engine = CeraEngine::from_parts(ModelBytes::text(model), cfg()).expect("from_parts");
    let mut session = engine
        .new_session(cera::SessionConfig::default())
        .expect("new_session");

    let err = session
        .append_image(&solid_png())
        .expect_err("a text bundle must refuse an image");
    assert!(
        matches!(err, cera::CeraError::UnsupportedModality),
        "expected UnsupportedModality, got {err:?}"
    );
}

/// Guard against the silent-skip failure mode: every test above returns early
/// when the bundle is missing, which reads as a pass. Setting
/// `CERA_REQUIRE_VL_FIXTURE=1` turns that into a hard failure so a CI job
/// that believes it is covering this path finds out when it is not.
#[test]
fn vl_fixture_is_present() {
    if std::env::var("CERA_REQUIRE_VL_FIXTURE").is_err() {
        return;
    }
    assert!(
        find_vl_bundle().is_some(),
        "CERA_REQUIRE_VL_FIXTURE is set but no LFM2-VL bundle was found under \
         ~/.leap/models; the from_parts tests would have silently skipped"
    );
}
