//! Error types and API error payload definitions for `cera-client`.

use serde::{Deserialize, Deserializer, Serialize};

/// Detailed error payload returned by OpenAI-compatible endpoints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiErrorPayload {
    /// Human-readable description of the error.
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

fn deserialize_string_or_int<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrInt {
        String(String),
        Int(i64),
    }

    let opt = Option::<StringOrInt>::deserialize(deserializer)?;
    Ok(opt.map(|val| match val {
        StringOrInt::String(s) => s,
        StringOrInt::Int(i) => i.to_string(),
    }))
}

/// JSON envelope containing the API error payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiErrorEnvelope {
    /// Inner error payload.
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
    #[error("JSON serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

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

    /// Returns true if the error is a transient server failure (HTTP 5xx or connection error).
    pub fn is_transient(&self) -> bool {
        if let Some(status) = self.status()
            && status.is_server_error()
        {
            return true;
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
    }
}
