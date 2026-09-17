#![cfg(not(target_arch = "wasm32"))]

#[path = "common/whisper_fixture.rs"]
mod fixture;

use cera_ffi::{
    FfiError, FfiWhisperModel, FfiWhisperTranscribeOpts, whisper_default_transcribe_opts,
};

fn options(max_tokens: u32) -> FfiWhisperTranscribeOpts {
    FfiWhisperTranscribeOpts {
        language: Some("en".into()),
        max_tokens: Some(max_tokens),
        ..Default::default()
    }
}

#[test]
fn options_errors_and_nonempty_decoding() {
    let defaults = whisper_default_transcribe_opts();
    assert_eq!(defaults, cera::WhisperTranscribeOpts::default().into());
    let custom = FfiWhisperTranscribeOpts {
        language: Some("es".into()),
        translate: true,
        timestamps: true,
        max_tokens: None,
        temperature: None,
    };
    let core: cera::WhisperTranscribeOpts = custom.into();
    assert_eq!(core.language.as_deref(), Some("es"));
    assert!(core.translate && core.timestamps);
    assert_eq!(core.max_tokens, 448);
    assert_eq!(core.temperature, 0.0);
    assert!(core.cancel.is_none());
    let capped: FfiWhisperTranscribeOpts = cera::WhisperTranscribeOpts {
        max_tokens: usize::MAX,
        ..Default::default()
    }
    .into();
    assert_eq!(capped.max_tokens, Some(u32::MAX));
    assert!(matches!(
        FfiWhisperModel::from_bytes(vec![1, 2]),
        Err(FfiError::Backend { .. })
    ));
    let missing =
        std::env::temp_dir().join(format!("cera-missing-whisper-{}.gguf", std::process::id()));
    assert!(!missing.exists());
    assert!(matches!(
        FfiWhisperModel::from_file(missing.to_string_lossy().into_owned()),
        Err(FfiError::Backend { .. })
    ));
    let model = FfiWhisperModel::from_bytes(fixture::model(false, true)).unwrap();
    assert!(model.is_multilingual());
    assert_eq!(model.languages().len(), 100);
    assert_eq!(&model.languages()[..2], ["en", "zh"]);
    assert_eq!(model.transcribe(Vec::new(), None).unwrap(), "");
    assert_eq!(
        model.transcribe(fixture::pcm(), Some(options(3))).unwrap(),
        "aaa"
    );
    assert_eq!(
        model.transcribe(fixture::pcm(), Some(options(0))).unwrap(),
        ""
    );
}

#[tokio::test]
async fn asynchronous_calls_keep_distinct_models_alive() {
    let a = FfiWhisperModel::from_bytes(fixture::model(false, true)).unwrap();
    let b = FfiWhisperModel::from_bytes(fixture::model(true, false)).unwrap();
    assert!(!b.is_multilingual());
    let pending_a = a.clone().transcribe_async(fixture::pcm(), Some(options(3)));
    let pending_b = b.clone().transcribe_async(fixture::pcm(), Some(options(2)));
    drop(a);
    drop(b);
    let (a, b) = tokio::join!(pending_a, pending_b);
    assert_eq!(a.unwrap(), "aaa");
    assert_eq!(b.unwrap(), "bb");
}

#[test]
fn file_and_byte_loaders_retain_their_weights() {
    let path = std::env::temp_dir().join(format!("cera-whisper-{}.gguf", std::process::id()));
    // create_new avoids overwriting any preexisting file, even after a failed run.
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let _cleanup = Cleanup(path.clone());
    use std::io::Write;
    file.write_all(&fixture::model(true, false)).unwrap();
    drop(file);
    let disk = FfiWhisperModel::from_file(path.to_str().unwrap().to_owned()).unwrap();
    #[cfg(unix)]
    std::fs::remove_file(&path).unwrap();
    let mut bytes = fixture::model(false, true);
    let memory = FfiWhisperModel::from_bytes(bytes.clone()).unwrap();
    bytes.fill(0);
    assert_eq!(
        disk.transcribe(fixture::pcm(), Some(options(2))).unwrap(),
        "bb"
    );
    assert_eq!(
        memory.transcribe(fixture::pcm(), Some(options(3))).unwrap(),
        "aaa"
    );
}

#[tokio::test]
async fn shared_model_calls_keep_decode_state_independent() {
    let model = FfiWhisperModel::from_bytes(fixture::model(false, true)).unwrap();
    let (a, b) = tokio::join!(
        model
            .clone()
            .transcribe_async(fixture::pcm(), Some(options(3))),
        model
            .clone()
            .transcribe_async(fixture::pcm(), Some(options(2))),
    );
    assert_eq!(a.unwrap(), "aaa");
    assert_eq!(b.unwrap(), "aa");
}

mod upstream_compatibility {
    use std::path::PathBuf;

    use cera_ffi::{FfiWhisperModel, FfiWhisperTranscribeOpts, whisper_default_transcribe_opts};

    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

    fn find_whisper_model() -> Option<PathBuf> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let candidates = [
            manifest_dir.join("../models/whisper.gguf"),
            manifest_dir.join("../models/whisper-tiny.en-Q5_K_M.gguf"),
            manifest_dir.join("../models/whisper-base.en-Q5_K_M.gguf"),
            PathBuf::from("models/whisper.gguf"),
        ];
        candidates.into_iter().find(|p| p.exists())
    }

    #[test]
    fn test_ffi_whisper_opts_and_defaults() {
        let def = whisper_default_transcribe_opts();
        assert_eq!(def.language, None);
        assert!(!def.translate);
        assert!(!def.timestamps);
        assert_eq!(def.max_tokens, Some(448));
        assert_eq!(def.temperature, Some(0.0));

        let custom = FfiWhisperTranscribeOpts {
            language: Some("fr".to_string()),
            translate: true,
            timestamps: true,
            max_tokens: Some(128),
            temperature: Some(0.5),
        };

        let core: cera::WhisperTranscribeOpts = custom.clone().into();
        assert_eq!(core.language.as_deref(), Some("fr"));
        assert!(core.translate);
        assert!(core.timestamps);
        assert_eq!(core.max_tokens, 128);
        assert_eq!(core.temperature, 0.5);

        let roundtrip: FfiWhisperTranscribeOpts = core.into();
        assert_eq!(roundtrip, custom);
    }

    #[test]
    fn test_ffi_whisper_error_handling_no_panic() {
        // 1. Loading non-existent file returns error without panicking
        let err_file = FfiWhisperModel::from_file("/path/to/nonexistent/whisper.gguf".to_string());
        assert!(err_file.is_err());

        // 2. Loading garbage bytes returns error without panicking
        let err_bytes = FfiWhisperModel::from_bytes(vec![0x47, 0x47, 0x55, 0x46, 0x00, 0x00]);
        assert!(err_bytes.is_err());
    }

    #[test]
    fn test_ffi_whisper_model_transcribe_when_available() -> Result<()> {
        let Some(model_path) = find_whisper_model() else {
            eprintln!("Skipping live model test: whisper GGUF not found");
            return Ok(());
        };

        let model_str = model_path.to_str().unwrap().to_string();
        let whisper = FfiWhisperModel::from_file(model_str)?;

        let languages = whisper.languages();
        assert_eq!(languages.len(), 100);
        assert_eq!(languages[0], "en");
        assert_eq!(languages[1], "zh");

        // Empty audio transcription must succeed with empty text
        let empty_text = whisper.transcribe(Vec::new(), None)?;
        assert_eq!(empty_text, "");

        Ok(())
    }
}
