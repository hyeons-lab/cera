//! Main HTTP client implementation for OpenAI and OpenRouter APIs.

use std::time::Duration;

use reqwest::Response;
use reqwest::header::HeaderMap;

use crate::error::{ApiErrorEnvelope, ClientError};
use crate::provider::{OPENAI_API_KEY_ENV, OPENAI_BASE_URL_ENV, OPENROUTER_API_KEY_ENV, Provider};
use crate::stream::ChatCompletionStream;
use crate::types::{
    ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse,
    ListModelsResponse,
};

/// Builder for constructing a configured [`Client`].
#[derive(Debug, Clone)]
pub struct ClientBuilder {
    provider: Provider,
    api_key: Option<String>,
    timeout: Option<Duration>,
    connect_timeout: Option<Duration>,
}

impl ClientBuilder {
    /// Create a new builder with the specified provider.
    pub fn new(provider: Provider) -> Self {
        Self {
            provider,
            api_key: None,
            timeout: Some(Duration::from_secs(60)),
            connect_timeout: None,
        }
    }

    /// Set the API key for authorization.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set request timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set socket connection timeout.
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = Some(timeout);
        self
    }

    /// Build the client instance.
    pub fn build(self) -> Result<Client, ClientError> {
        #[allow(unused_mut)]
        let mut http_builder = reqwest::Client::builder();
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(ct) = self.connect_timeout {
                http_builder = http_builder.connect_timeout(ct);
            }
            http_builder =
                http_builder.user_agent(concat!("cera-client/", env!("CARGO_PKG_VERSION")));
        }

        let http = http_builder.build()?;
        Ok(Client {
            http,
            provider: self.provider,
            api_key: self.api_key,
            timeout: self.timeout,
        })
    }
}

/// Asynchronous API client for querying OpenAI and OpenRouter endpoints.
#[derive(Debug, Clone)]
pub struct Client {
    http: reqwest::Client,
    provider: Provider,
    api_key: Option<String>,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    timeout: Option<Duration>,
}

impl Client {
    /// Create a new client targeting OpenAI with an API key.
    pub fn new(api_key: impl Into<String>) -> Result<Self, ClientError> {
        Self::openai(api_key)
    }

    /// Create a client targeting the official OpenAI API.
    pub fn openai(api_key: impl Into<String>) -> Result<Self, ClientError> {
        Self::builder(Provider::OpenAi).api_key(api_key).build()
    }

    /// Create a client targeting the OpenRouter API.
    pub fn openrouter(api_key: impl Into<String>) -> Result<Self, ClientError> {
        Self::builder(Provider::openrouter())
            .api_key(api_key)
            .build()
    }

    /// Create a client targeting OpenRouter with application attribution metadata.
    pub fn openrouter_with_attribution(
        api_key: impl Into<String>,
        app_url: impl Into<String>,
        app_name: impl Into<String>,
    ) -> Result<Self, ClientError> {
        Self::builder(Provider::openrouter_with_attribution(app_url, app_name))
            .api_key(api_key)
            .build()
    }

    /// Create a client targeting a custom OpenAI-compatible server.
    pub fn custom(
        base_url: impl Into<String>,
        api_key: Option<String>,
    ) -> Result<Self, ClientError> {
        let mut b = Self::builder(Provider::custom(base_url));
        if let Some(key) = api_key {
            b = b.api_key(key);
        }
        b.build()
    }

    /// Automatically configure a client from available environment variables.
    ///
    /// Checks `OPENROUTER_API_KEY` first; if present, connects to OpenRouter.
    /// Then checks `OPENAI_BASE_URL`; if present, connects to that custom endpoint
    /// with optional `OPENAI_API_KEY`.
    /// Otherwise checks `OPENAI_API_KEY` to connect to OpenAI.
    pub fn from_env() -> Result<Self, ClientError> {
        if let Ok(key) = std::env::var(OPENROUTER_API_KEY_ENV)
            && !key.trim().is_empty()
        {
            return Self::openrouter(key);
        }

        if let Ok(base_url) = std::env::var(OPENAI_BASE_URL_ENV)
            && !base_url.trim().is_empty()
        {
            let key = std::env::var(OPENAI_API_KEY_ENV)
                .ok()
                .filter(|k| !k.trim().is_empty());
            return Self::custom(base_url, key);
        }

        if let Ok(key) = std::env::var(OPENAI_API_KEY_ENV)
            && !key.trim().is_empty()
        {
            return Self::openai(key);
        }

        Err(ClientError::MissingApiKey(format!(
            "Neither {OPENROUTER_API_KEY_ENV}, {OPENAI_BASE_URL_ENV}, nor {OPENAI_API_KEY_ENV} is set in the environment"
        )))
    }

    /// Initialize a builder with a specific provider.
    pub fn builder(provider: Provider) -> ClientBuilder {
        ClientBuilder::new(provider)
    }

    /// Returns a reference to the active provider.
    pub fn provider(&self) -> &Provider {
        &self.provider
    }

    /// Returns the configured API key if any.
    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    /// Sends a non-streaming chat completion request to `/chat/completions`.
    pub async fn chat(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ClientError> {
        request.stream = Some(false);
        let url = self.provider.endpoint_url("chat/completions")?;
        let mut headers = HeaderMap::new();
        self.provider
            .apply_headers(&mut headers, self.api_key.as_deref())?;

        #[allow(unused_mut)]
        let mut req = self.http.post(url).headers(headers).json(&request);
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(to) = self.timeout {
            req = req.timeout(to);
        }
        let response = req.send().await?;

        let checked = Self::check_response(response).await?;
        let body = checked.json::<ChatCompletionResponse>().await?;
        Ok(body)
    }

    /// Sends a streaming chat completion request to `/chat/completions`, returning an SSE chunk stream.
    ///
    /// The request handshake awaits HTTP response headers. Once headers arrive and status is
    /// validated, the connection remains open for streaming tokens. Total request timeout is omitted
    /// to support long generations; connection timeout is governed by [`ClientBuilder::connect_timeout`].
    pub async fn chat_stream(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Result<
        ChatCompletionStream<
            impl futures_core::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin,
        >,
        ClientError,
    > {
        request.stream = Some(true);
        let url = self.provider.endpoint_url("chat/completions")?;
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::ACCEPT,
            reqwest::header::HeaderValue::from_static("text/event-stream"),
        );
        headers.insert(
            reqwest::header::CACHE_CONTROL,
            reqwest::header::HeaderValue::from_static("no-cache"),
        );
        self.provider
            .apply_headers(&mut headers, self.api_key.as_deref())?;

        let response = self
            .http
            .post(url)
            .headers(headers)
            .json(&request)
            .send()
            .await?;

        let checked = Self::check_response(response).await?;
        let content_type = checked
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|ct| ct.to_str().ok())
            .map(|s| s.trim().to_lowercase());

        if let Some(ct_lower) = content_type
            && !ct_lower.starts_with("text/event-stream")
            && !ct_lower.starts_with("application/x-ndjson")
        {
            let status = checked.status();
            #[cfg(not(target_arch = "wasm32"))]
            let text = {
                // Read at most 64 KB to avoid unbounded memory usage on unexpected non-SSE streams
                let mut body_bytes = Vec::new();
                let mut checked = checked;
                while let Ok(Some(chunk)) = checked.chunk().await {
                    let remaining = 64 * 1024 - body_bytes.len();
                    if chunk.len() <= remaining {
                        body_bytes.extend_from_slice(&chunk);
                    } else {
                        body_bytes.extend_from_slice(&chunk[..remaining]);
                        break;
                    }
                    if body_bytes.len() >= 64 * 1024 {
                        break;
                    }
                }
                String::from_utf8(body_bytes)
                    .unwrap_or_else(|err| String::from_utf8_lossy(err.as_bytes()).into_owned())
            };
            #[cfg(target_arch = "wasm32")]
            let text = checked.text().await.unwrap_or_default();
            if let Ok(envelope) = serde_json::from_str::<ApiErrorEnvelope>(&text) {
                return Err(ClientError::Api {
                    status: Some(status),
                    message: envelope.error.message,
                    error_type: envelope.error.error_type,
                    code: envelope.error.code,
                    param: envelope.error.param,
                });
            }
            let preview = if text.len() > 512 {
                let mut end = 512;
                while !text.is_char_boundary(end) {
                    end -= 1;
                }
                format!("{}... [truncated]", &text[..end])
            } else {
                text
            };
            return Err(ClientError::Stream(format!(
                "Expected text/event-stream content type, received {ct_lower}: {preview}"
            )));
        }

        let stream = checked.bytes_stream();
        Ok(ChatCompletionStream::new(stream))
    }

    /// Generates vector embeddings for input text via `/embeddings`.
    pub async fn embeddings(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, ClientError> {
        let url = self.provider.endpoint_url("embeddings")?;
        let mut headers = HeaderMap::new();
        self.provider
            .apply_headers(&mut headers, self.api_key.as_deref())?;

        #[allow(unused_mut)]
        let mut req = self.http.post(url).headers(headers).json(&request);
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(to) = self.timeout {
            req = req.timeout(to);
        }
        let response = req.send().await?;

        let checked = Self::check_response(response).await?;
        let body = checked.json::<EmbeddingResponse>().await?;
        Ok(body)
    }

    /// Lists models available from the provider via `/models`.
    pub async fn models(&self) -> Result<ListModelsResponse, ClientError> {
        let url = self.provider.endpoint_url("models")?;
        let mut headers = HeaderMap::new();
        self.provider
            .apply_headers(&mut headers, self.api_key.as_deref())?;

        #[allow(unused_mut)]
        let mut req = self.http.get(url).headers(headers);
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(to) = self.timeout {
            req = req.timeout(to);
        }
        let response = req.send().await?;

        let checked = Self::check_response(response).await?;
        let body = checked.json::<ListModelsResponse>().await?;
        Ok(body)
    }

    /// Validates response status, parsing error payloads if unsuccessful.
    async fn check_response(response: Response) -> Result<Response, ClientError> {
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }

        let raw_text = response.text().await.unwrap_or_default();
        if let Ok(envelope) = serde_json::from_str::<ApiErrorEnvelope>(&raw_text) {
            return Err(ClientError::Api {
                status: Some(status),
                message: envelope.error.message,
                error_type: envelope.error.error_type,
                code: envelope.error.code,
                param: envelope.error.param,
            });
        }

        Err(ClientError::Api {
            status: Some(status),
            message: if raw_text.is_empty() {
                format!("HTTP error {status}")
            } else {
                raw_text
            },
            error_type: None,
            code: None,
            param: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_constructors() {
        let client_oa = Client::openai("sk-12345").unwrap();
        assert_eq!(client_oa.provider(), &Provider::OpenAi);
        assert_eq!(client_oa.api_key(), Some("sk-12345"));

        let client_or = Client::openrouter("or-67890").unwrap();
        assert_eq!(client_or.provider(), &Provider::openrouter());
        assert_eq!(client_or.api_key(), Some("or-67890"));

        let client_custom = Client::custom("http://localhost:11434/v1", None).unwrap();
        assert_eq!(
            client_custom.provider(),
            &Provider::custom("http://localhost:11434/v1")
        );
        assert_eq!(client_custom.api_key(), None);
    }

    #[test]
    fn test_client_builder_customization() {
        let client = Client::builder(Provider::openrouter_with_attribution(
            "https://test.com",
            "TestApp",
        ))
        .api_key("key-abc")
        .timeout(Duration::from_secs(120))
        .connect_timeout(Duration::from_secs(10))
        .build()
        .unwrap();

        assert_eq!(client.api_key(), Some("key-abc"));
        match client.provider() {
            Provider::OpenRouter { app_url, app_name } => {
                assert_eq!(app_url.as_deref(), Some("https://test.com"));
                assert_eq!(app_name.as_deref(), Some("TestApp"));
            }
            _ => panic!("expected OpenRouter provider"),
        }
    }
}
