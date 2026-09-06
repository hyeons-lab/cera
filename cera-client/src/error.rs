//! Error types and API error payload definitions for `cera-client`.

use serde::{Deserialize, Deserializer, Serialize};

/// Detailed error payload returned by OpenAI-compatible endpoints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiErrorPayload {
    /// Human-readable description of the error.
    #[serde(default, deserialize_with = "deserialize_null_as_default")]
    pub message: String,
    /// Error category or classification (e.g. `invalid_request_error`).
    #[serde(rename = "type", default)]
    pub error_type: Option<String>,
    /// Specific parameter that caused the error, if applicable.
    #[serde(default)]
    pub param: Option<String>,
    /// Machine-readable error code (e.g. `rate_limit_exceeded` or OpenRouter numeric codes).
    #[serde(default, deserialize_with = "deserialize_string_or_int")]
    pub code: Option<String>,
}

fn deserialize_null_as_default<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let opt = Option::<String>::deserialize(deserializer)?;
    Ok(opt.unwrap_or_default())
}

fn deserialize_string_or_int<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let opt = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(opt.and_then(|val| match val {
        serde_json::Value::String(s) => Some(s),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        serde_json::Value::Null => None,
        other => Some(other.to_string()),
    }))
}

fn deserialize_api_error_payload<'de, D>(deserializer: D) -> Result<ApiErrorPayload, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum ErrorPayloadOrString {
        Payload(ApiErrorPayload),
        String(String),
    }

    let val = ErrorPayloadOrString::deserialize(deserializer)?;
    Ok(match val {
        ErrorPayloadOrString::Payload(p) => p,
        ErrorPayloadOrString::String(s) => ApiErrorPayload {
            message: s,
            error_type: None,
            param: None,
            code: None,
        },
    })
}

/// JSON envelope containing the API error payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiErrorEnvelope {
    /// Inner error payload.
    #[serde(deserialize_with = "deserialize_api_error_payload")]
    pub error: ApiErrorPayload,
}

/// Errors that can occur when calling OpenAI and OpenRouter endpoints.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    /// Network or HTTP transport failure.
    #[error("HTTP transport error: {0}")]
    Http(#[from] reqwest::Error),

    /// The remote endpoint returned an error status code and payload.
    #[error("API error{}: {message}", .status.map(|s| format!(" (status {s})")).unwrap_or_default())]
    Api {
        /// HTTP response status code, if available.
        status: Option<reqwest::StatusCode>,
        /// Human-readable error message.
        message: String,
        /// Error classification string.
        error_type: Option<String>,
        /// Machine-readable error code.
        code: Option<String>,
        /// Parameter that caused the error.
        param: Option<String>,
    },

    /// JSON serialization or deserialization failure.
    #[error(
        "JSON serialization error: {source}{}",
        .raw_payload.as_deref().map(|p| format!(" (raw payload: {p})")).unwrap_or_default()
    )]
    Serialization {
        /// Underlying serde_json error.
        #[source]
        source: serde_json::Error,
        /// Raw unparsed payload if available for diagnostics.
        raw_payload: Option<String>,
    },

    /// Server-Sent Events streaming error or protocol violation.
    #[error("Stream error: {0}")]
    Stream(String),

    /// An API key was required but not provided or found in the environment.
    #[error("Missing API key: {0}")]
    MissingApiKey(String),

    /// The provided base URL or endpoint could not be parsed.
    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    /// An HTTP header value contains invalid characters.
    #[error("Invalid HTTP header: {0}")]
    InvalidHeader(String),
}

impl From<serde_json::Error> for ClientError {
    fn from(source: serde_json::Error) -> Self {
        Self::Serialization {
            source,
            raw_payload: None,
        }
    }
}

impl ClientError {
    /// Returns the HTTP status code if this error originated from an API response.
    pub fn status(&self) -> Option<reqwest::StatusCode> {
        match self {
            Self::Api { status, .. } => *status,
            Self::Http(err) => err.status(),
            _ => None,
        }
    }

    /// Returns true if the error represents an HTTP 429 Too Many Requests response or rate limit code.
    pub fn is_rate_limited(&self) -> bool {
        if self.status() == Some(reqwest::StatusCode::TOO_MANY_REQUESTS) {
            return true;
        }
        if let Self::Api { code, .. } = self
            && let Some(c) = code.as_deref()
        {
            return c == "429" || c == "rate_limit_exceeded" || c == "rate_limit";
        }
        false
    }

    /// Returns true if the error is a transient server failure (HTTP 5xx, 408, or connection error).
    pub fn is_transient(&self) -> bool {
        if let Some(status) = self.status()
            && (status.is_server_error() || status == reqwest::StatusCode::REQUEST_TIMEOUT)
        {
            return true;
        }
        if let Self::Api {
            code, error_type, ..
        } = self
        {
            if let Some(c) = code.as_deref()
                && matches!(
                    c,
                    "500"
                        | "502"
                        | "503"
                        | "504"
                        | "server_error"
                        | "overloaded"
                        | "timeout"
                        | "service_unavailable"
                        | "gateway_timeout"
                        | "engine_overloaded"
                )
            {
                return true;
            }
            if let Some(t) = error_type.as_deref()
                && (t == "server_error" || t == "timeout" || t == "service_unavailable")
            {
                return true;
            }
        }
        if let Self::Http(err) = self {
            #[cfg(not(target_arch = "wasm32"))]
            return err.is_timeout() || err.is_connect();
            #[cfg(target_arch = "wasm32")]
            return err.is_timeout();
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_string_error_code() {
        let json = r#"{"error": {"message": "Rate limit exceeded", "type": "tokens", "code": "rate_limit_exceeded"}}"#;
        let env: ApiErrorEnvelope = serde_json::from_str(json).unwrap();
        assert_eq!(env.error.code.as_deref(), Some("rate_limit_exceeded"));
        assert_eq!(env.error.message, "Rate limit exceeded");
    }

    #[test]
    fn test_parse_numeric_error_code() {
        let json = r#"{"error": {"message": "Payment required", "code": 402}}"#;
        let env: ApiErrorEnvelope = serde_json::from_str(json).unwrap();
        assert_eq!(env.error.code.as_deref(), Some("402"));
        assert_eq!(env.error.message, "Payment required");
    }

    #[test]
    fn test_error_status_and_helpers() {
        let err429 = ClientError::Api {
            status: Some(reqwest::StatusCode::TOO_MANY_REQUESTS),
            message: "Rate limit reached".to_string(),
            error_type: None,
            code: None,
            param: None,
        };
        assert_eq!(
            err429.status(),
            Some(reqwest::StatusCode::TOO_MANY_REQUESTS)
        );
        assert!(err429.is_rate_limited());
        assert!(!err429.is_transient());

        let err_code_rate_limit = ClientError::Api {
            status: None,
            message: "Too fast".to_string(),
            error_type: None,
            code: Some("rate_limit_exceeded".to_string()),
            param: None,
        };
        assert_eq!(err_code_rate_limit.status(), None);
        assert!(err_code_rate_limit.is_rate_limited());
        assert!(!err_code_rate_limit.is_transient());

        let err500 = ClientError::Api {
            status: Some(reqwest::StatusCode::INTERNAL_SERVER_ERROR),
            message: "Internal server error".to_string(),
            error_type: None,
            code: None,
            param: None,
        };
        assert_eq!(
            err500.status(),
            Some(reqwest::StatusCode::INTERNAL_SERVER_ERROR)
        );
        assert!(!err500.is_rate_limited());
        assert!(err500.is_transient());

        let err_in_band_503 = ClientError::Api {
            status: None,
            message: "Model overloaded".to_string(),
            error_type: Some("server_error".to_string()),
            code: Some("503".to_string()),
            param: None,
        };
        assert!(err_in_band_503.is_transient());

        let err408 = ClientError::Api {
            status: Some(reqwest::StatusCode::REQUEST_TIMEOUT),
            message: "Timeout".to_string(),
            error_type: None,
            code: None,
            param: None,
        };
        assert!(err408.is_transient());

        let err_service_unavailable = ClientError::Api {
            status: None,
            message: "Service unavailable".to_string(),
            error_type: Some("service_unavailable".to_string()),
            code: Some("service_unavailable".to_string()),
            param: None,
        };
        assert!(err_service_unavailable.is_transient());

        let err_gateway_timeout = ClientError::Api {
            status: None,
            message: "Gateway timeout".to_string(),
            error_type: None,
            code: Some("gateway_timeout".to_string()),
            param: None,
        };
        assert!(err_gateway_timeout.is_transient());
    }

    #[test]
    fn test_parse_null_or_missing_error_message() {
        let json_null = r#"{"error": {"message": null, "code": 500}}"#;
        let env_null: ApiErrorEnvelope = serde_json::from_str(json_null).unwrap();
        assert_eq!(env_null.error.message, "");
        assert_eq!(env_null.error.code.as_deref(), Some("500"));

        let json_missing = r#"{"error": {"code": 503}}"#;
        let env_missing: ApiErrorEnvelope = serde_json::from_str(json_missing).unwrap();
        assert_eq!(env_missing.error.message, "");
        assert_eq!(env_missing.error.code.as_deref(), Some("503"));
    }

    #[test]
    fn test_parse_bare_string_error_message() {
        let json_str = r#"{"error": "model is currently overloaded, please try again"}"#;
        let env: ApiErrorEnvelope = serde_json::from_str(json_str).unwrap();
        assert_eq!(
            env.error.message,
            "model is currently overloaded, please try again"
        );
        assert_eq!(env.error.code, None);
        assert_eq!(env.error.error_type, None);
    }

    #[test]
    fn test_parse_various_error_codes() {
        let json_float = r#"{"error": {"message": "Rate limit", "code": 429.5}}"#;
        let env_float: ApiErrorEnvelope = serde_json::from_str(json_float).unwrap();
        assert_eq!(env_float.error.code.as_deref(), Some("429.5"));

        let json_bool = r#"{"error": {"message": "Invalid", "code": true}}"#;
        let env_bool: ApiErrorEnvelope = serde_json::from_str(json_bool).unwrap();
        assert_eq!(env_bool.error.code.as_deref(), Some("true"));

        let json_null_code = r#"{"error": {"message": "Error", "code": null}}"#;
        let env_null_code: ApiErrorEnvelope = serde_json::from_str(json_null_code).unwrap();
        assert_eq!(env_null_code.error.code, None);
    }

    #[test]
    fn test_serialization_error_display_with_and_without_raw_payload() {
        let err_no_payload: ClientError =
            serde_json::from_str::<serde_json::Value>("invalid json{")
                .unwrap_err()
                .into();
        let display_no_payload = err_no_payload.to_string();
        assert!(display_no_payload.starts_with("JSON serialization error:"));
        assert!(!display_no_payload.contains("raw payload"));

        let err_with_payload = ClientError::Serialization {
            source: serde_json::from_str::<serde_json::Value>("invalid json{").unwrap_err(),
            raw_payload: Some("data: {malformed}".to_string()),
        };
        let display_with_payload = err_with_payload.to_string();
        assert!(display_with_payload.contains("raw payload: data: {malformed}"));
    }
}
