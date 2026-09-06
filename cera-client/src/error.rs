//! Error types and API error payload definitions for `cera-client`.

use serde::{Deserialize, Serialize};

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
    /// Machine-readable error code (e.g. `rate_limit_exceeded`).
    #[serde(default)]
    pub code: Option<String>,
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
    #[error("API error (status {status}): {message}")]
    Api {
        /// HTTP response status code.
        status: reqwest::StatusCode,
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
}

impl ClientError {
    /// Returns the HTTP status code if this error originated from an API response.
    pub fn status(&self) -> Option<reqwest::StatusCode> {
        match self {
            Self::Api { status, .. } => Some(*status),
            Self::Http(err) => err.status(),
            _ => None,
        }
    }

    /// Returns true if the error represents an HTTP 429 Too Many Requests response.
    pub fn is_rate_limited(&self) -> bool {
        self.status() == Some(reqwest::StatusCode::TOO_MANY_REQUESTS)
    }

    /// Returns true if the error is a transient server failure (HTTP 5xx or connection error).
    pub fn is_transient(&self) -> bool {
        if let Some(status) = self.status()
            && status.is_server_error()
        {
            return true;
        }
        if let Self::Http(err) = self {
            return err.is_timeout() || err.is_connect();
        }
        false
    }
}
