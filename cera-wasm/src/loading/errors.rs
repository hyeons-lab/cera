use cera as core;
use wasm_bindgen::prelude::*;

#[derive(Debug, thiserror::Error)]
pub(super) enum LoadError {
    #[error("expected {expected}, found {actual}: {architecture}")]
    KindMismatch {
        expected: String,
        actual: String,
        architecture: String,
    },
    #[error("unsupported architecture: {architecture}")]
    UnsupportedArchitecture { architecture: String },
    #[error("unsupported inference type: {inference_type}")]
    UnsupportedInferenceType { inference_type: String },
    #[error("{detail}")]
    Source { source_kind: String, detail: String },
    #[error("{detail}")]
    Assembly { backend: String, detail: String },
    #[error("{detail}")]
    InvalidConfig {
        field: String,
        value: String,
        reason: String,
        detail: String,
    },
    #[error("{detail}")]
    Engine { detail: String },
    #[error("loader already consumed")]
    Consumed,
}

impl From<core::LoadError> for LoadError {
    fn from(error: core::LoadError) -> Self {
        match error {
            core::LoadError::KindMismatch {
                expected,
                actual,
                architecture,
            } => Self::KindMismatch {
                expected: format!("{expected:?}"),
                actual: format!("{actual:?}"),
                architecture,
            },
            core::LoadError::UnsupportedArchitecture { architecture } => {
                Self::UnsupportedArchitecture { architecture }
            }
            core::LoadError::Source {
                error: cera::CeraError::UnsupportedInferenceType(inference_type),
                ..
            }
            | core::LoadError::Assembly {
                error: cera::CeraError::UnsupportedInferenceType(inference_type),
                ..
            }
            | core::LoadError::Engine(cera::CeraError::UnsupportedInferenceType(inference_type)) => {
                Self::UnsupportedInferenceType { inference_type }
            }
            core::LoadError::Source { source_kind, error } => Self::Source {
                source_kind: source_kind.into(),
                detail: error.to_string(),
            },
            core::LoadError::Assembly { backend, error } => Self::Assembly {
                backend: format!("{backend:?}"),
                detail: error.to_string(),
            },
            other => Self::Engine {
                detail: other.to_string(),
            },
        }
    }
}

pub(super) fn error(error: LoadError) -> JsValue {
    let object = js_sys::Error::new(&error.to_string());
    let mut fields = Vec::new();
    let code = match error {
        LoadError::KindMismatch {
            expected,
            actual,
            architecture,
        } => {
            fields.extend([
                ("expected", expected),
                ("actual", actual),
                ("architecture", architecture),
            ]);
            "KindMismatch"
        }
        LoadError::UnsupportedArchitecture { architecture } => {
            fields.push(("architecture", architecture));
            "UnsupportedArchitecture"
        }
        LoadError::UnsupportedInferenceType { inference_type } => {
            fields.push(("inference_type", inference_type));
            "UnsupportedInferenceType"
        }
        LoadError::Source { source_kind, .. } => {
            fields.push(("source_kind", source_kind));
            "Source"
        }
        LoadError::Assembly { backend, .. } => {
            fields.push(("backend", backend));
            "Assembly"
        }
        LoadError::InvalidConfig {
            field,
            value,
            reason,
            ..
        } => {
            fields.extend([("field", field), ("value", value), ("reason", reason)]);
            "InvalidConfig"
        }
        LoadError::Engine { .. } => "Engine",
        LoadError::Consumed => "Consumed",
    };
    fields.push(("code", code.into()));
    for (key, value) in fields {
        js_sys::Reflect::set(&object, &key.into(), &value.into()).expect("new error properties");
    }
    object.into()
}
