//! Provider definitions and URL / header resolution for remote endpoints.

use reqwest::Url;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};

use crate::error::ClientError;

/// Default base URL for the OpenAI API.
pub const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Default base URL for the OpenRouter API.
pub const OPENROUTER_DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Environment variable for OpenAI API key.
pub const OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";

/// Environment variable for OpenRouter API key.
pub const OPENROUTER_API_KEY_ENV: &str = "OPENROUTER_API_KEY";

/// Environment variable for custom OpenAI base URL.
pub const OPENAI_BASE_URL_ENV: &str = "OPENAI_BASE_URL";

/// Endpoint provider target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Provider {
    /// Official OpenAI API endpoint (`https://api.openai.com/v1`).
    OpenAi,

    /// OpenRouter API endpoint (`https://openrouter.ai/api/v1`) with optional attribution headers.
    OpenRouter {
        /// Optional site URL sent as `HTTP-Referer` for OpenRouter rankings.
        app_url: Option<String>,
        /// Optional site name sent as `X-Title` for OpenRouter rankings.
        app_name: Option<String>,
    },

    /// Any custom OpenAI-compatible server (e.g. vLLM, Ollama, Groq, Together, DeepSeek).
    Custom {
        /// Base URL for the API (e.g. `http://localhost:8000/v1`).
        base_url: String,
    },
}

impl Provider {
    /// Create an OpenRouter provider with default settings.
    pub fn openrouter() -> Self {
        Self::OpenRouter {
            app_url: None,
            app_name: None,
        }
    }

    /// Create an OpenRouter provider with site URL and application name for OpenRouter rankings.
    pub fn openrouter_with_attribution(
        app_url: impl Into<String>,
        app_name: impl Into<String>,
    ) -> Self {
        Self::OpenRouter {
            app_url: Some(app_url.into()),
            app_name: Some(app_name.into()),
        }
    }

    /// Create a custom OpenAI-compatible provider with a specific base URL.
    pub fn custom(base_url: impl Into<String>) -> Self {
        Self::Custom {
            base_url: base_url.into(),
        }
    }

    /// Returns the base URL string for this provider.
    pub fn base_url(&self) -> &str {
        match self {
            Self::OpenAi => OPENAI_DEFAULT_BASE_URL,
            Self::OpenRouter { .. } => OPENROUTER_DEFAULT_BASE_URL,
            Self::Custom { base_url } => base_url.as_str(),
        }
    }

    /// Returns the canonical environment variable name for this provider's API key.
    pub fn default_env_var(&self) -> Option<&'static str> {
        match self {
            Self::OpenAi => Some(OPENAI_API_KEY_ENV),
            Self::OpenRouter { .. } => Some(OPENROUTER_API_KEY_ENV),
            Self::Custom { .. } => None,
        }
    }

    /// Constructs a full URL for a given relative path (e.g. `/chat/completions`).
    pub fn endpoint_url(&self, path: &str) -> Result<Url, ClientError> {
        let base = self.base_url().trim_end_matches('/');
        let subpath = path.trim_start_matches('/');
        let full = format!("{base}/{subpath}");
        Url::parse(&full).map_err(|e| ClientError::InvalidUrl(format!("{full}: {e}")))
    }

    /// Injects authentication and provider-specific headers into a request header map.
    pub fn apply_headers(
        &self,
        headers: &mut HeaderMap,
        api_key: Option<&str>,
    ) -> Result<(), ClientError> {
        if let Some(key) = api_key {
            let auth_value = format!("Bearer {key}");
            let mut val = HeaderValue::from_str(&auth_value)
                .map_err(|e| ClientError::InvalidHeader(format!("Invalid auth header: {e}")))?;
            val.set_sensitive(true);
            headers.insert(AUTHORIZATION, val);
        }

        if let Self::OpenRouter { app_url, app_name } = self {
            if let Some(url) = app_url {
                match HeaderValue::from_str(url) {
                    Ok(val) => {
                        headers.insert(HeaderName::from_static("http-referer"), val);
                    }
                    Err(e) => {
                        tracing::warn!(target: "cera_client", error = %e, "Invalid http-referer header value; omitting");
                    }
                }
            }
            if let Some(name) = app_name {
                match HeaderValue::from_str(name) {
                    Ok(val) => {
                        headers.insert(HeaderName::from_static("x-title"), val);
                    }
                    Err(e) => {
                        tracing::warn!(target: "cera_client", error = %e, "Invalid x-title header value; omitting");
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_endpoints_and_headers() {
        let provider = Provider::OpenAi;
        assert_eq!(provider.base_url(), "https://api.openai.com/v1");
        assert_eq!(provider.default_env_var(), Some("OPENAI_API_KEY"));

        let url = provider.endpoint_url("chat/completions").unwrap();
        assert_eq!(url.as_str(), "https://api.openai.com/v1/chat/completions");

        let mut headers = HeaderMap::new();
        provider
            .apply_headers(&mut headers, Some("sk-test123"))
            .unwrap();
        assert_eq!(
            headers.get(AUTHORIZATION).unwrap().to_str().unwrap(),
            "Bearer sk-test123"
        );
        assert!(!headers.contains_key("http-referer"));
    }

    #[test]
    fn test_openrouter_endpoints_and_attribution_headers() {
        let provider = Provider::openrouter_with_attribution("https://myapp.example.com", "MyApp");
        assert_eq!(provider.base_url(), "https://openrouter.ai/api/v1");
        assert_eq!(provider.default_env_var(), Some("OPENROUTER_API_KEY"));

        let url = provider.endpoint_url("/embeddings").unwrap();
        assert_eq!(url.as_str(), "https://openrouter.ai/api/v1/embeddings");

        let mut headers = HeaderMap::new();
        provider
            .apply_headers(&mut headers, Some("or-key-abc"))
            .unwrap();
        assert_eq!(
            headers.get(AUTHORIZATION).unwrap().to_str().unwrap(),
            "Bearer or-key-abc"
        );
        assert_eq!(
            headers.get("http-referer").unwrap().to_str().unwrap(),
            "https://myapp.example.com"
        );
        assert_eq!(headers.get("x-title").unwrap().to_str().unwrap(), "MyApp");
    }

    #[test]
    fn test_custom_provider() {
        let provider = Provider::custom("http://127.0.0.1:8000/v1/");
        assert_eq!(provider.base_url(), "http://127.0.0.1:8000/v1/");
        assert_eq!(provider.default_env_var(), None);

        let url = provider.endpoint_url("models").unwrap();
        assert_eq!(url.as_str(), "http://127.0.0.1:8000/v1/models");

        let mut headers = HeaderMap::new();
        provider.apply_headers(&mut headers, None).unwrap();
        assert!(!headers.contains_key(AUTHORIZATION));
    }

    #[test]
    fn test_invalid_auth_header() {
        let provider = Provider::OpenAi;
        let mut headers = HeaderMap::new();
        let err = provider
            .apply_headers(&mut headers, Some("key_with_\ninvalid_newline"))
            .unwrap_err();
        match err {
            ClientError::InvalidHeader(msg) => assert!(msg.contains("Invalid auth header")),
            other => panic!("expected InvalidHeader error, got {other:?}"),
        }
    }
}
