//! Main HTTP client implementation for OpenAI and OpenRouter APIs.

use std::time::Duration;

use reqwest::Response;
use reqwest::header::HeaderMap;

use crate::error::{ApiErrorEnvelope, ClientError};
use crate::provider::{OPENAI_API_KEY_ENV, OPENROUTER_API_KEY_ENV, Provider};
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
}

impl ClientBuilder {
    /// Create a new builder with the specified provider.
    pub fn new(provider: Provider) -> Self {
        Self {
            provider,
            api_key: None,
            timeout: Some(Duration::from_secs(60)),
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

    /// Build the client instance.
    pub fn build(self) -> Result<Client, ClientError> {
        let mut http_builder = reqwest::Client::builder();
        if let Some(to) = self.timeout {
            http_builder = http_builder.timeout(to);
        }

        let http = http_builder.build()?;
        Ok(Client {
            http,
            provider: self.provider,
            api_key: self.api_key,
        })
    }
}

/// Asynchronous API client for querying OpenAI and OpenRouter endpoints.
#[derive(Debug, Clone)]
pub struct Client {
    http: reqwest::Client,
    provider: Provider,
    api_key: Option<String>,
}

impl Client {
    /// Create a new client targeting OpenAI with an API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::openai(api_key)
    }

    /// Create a client targeting the official OpenAI API.
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self::builder(Provider::OpenAi)
            .api_key(api_key)
            .build()
            .expect("default HTTP client configuration should not fail")
    }

    /// Create a client targeting the OpenRouter API.
    pub fn openrouter(api_key: impl Into<String>) -> Self {
        Self::builder(Provider::openrouter())
            .api_key(api_key)
            .build()
            .expect("default HTTP client configuration should not fail")
    }

    /// Create a client targeting OpenRouter with application attribution metadata.
    pub fn openrouter_with_attribution(
        api_key: impl Into<String>,
        app_url: impl Into<String>,
        app_name: impl Into<String>,
    ) -> Self {
        Self::builder(Provider::openrouter_with_attribution(app_url, app_name))
            .api_key(api_key)
            .build()
            .expect("default HTTP client configuration should not fail")
    }

    /// Create a client targeting a custom OpenAI-compatible server.
    pub fn custom(base_url: impl Into<String>, api_key: Option<String>) -> Self {
        let mut b = Self::builder(Provider::custom(base_url));
        if let Some(key) = api_key {
            b = b.api_key(key);
        }
        b.build()
            .expect("default HTTP client configuration should not fail")
    }

    /// Automatically configure a client from available environment variables.
    ///
    /// Checks `OPENROUTER_API_KEY` first; if present, connects to OpenRouter.
    /// Otherwise checks `OPENAI_API_KEY` to connect to OpenAI.
    pub fn from_env() -> Result<Self, ClientError> {
        if let Ok(key) = std::env::var(OPENROUTER_API_KEY_ENV)
            && !key.trim().is_empty()
        {
            return Ok(Self::openrouter(key));
        }

        if let Ok(key) = std::env::var(OPENAI_API_KEY_ENV)
            && !key.trim().is_empty()
        {
            return Ok(Self::openai(key));
        }

        Err(ClientError::MissingApiKey(format!(
            "Neither {OPENROUTER_API_KEY_ENV} nor {OPENAI_API_KEY_ENV} is set in the environment"
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
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ClientError> {
        let url = self.provider.endpoint_url("chat/completions")?;
        let mut headers = HeaderMap::new();
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
        let body = checked.json::<ChatCompletionResponse>().await?;
        Ok(body)
    }

    /// Sends a streaming chat completion request to `/chat/completions`, returning an SSE chunk stream.
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

        let response = self
            .http
            .post(url)
            .headers(headers)
            .json(&request)
            .send()
            .await?;

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

        let response = self.http.get(url).headers(headers).send().await?;

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
                status,
                message: envelope.error.message,
                error_type: envelope.error.error_type,
                code: envelope.error.code,
                param: envelope.error.param,
            });
        }

        Err(ClientError::Api {
            status,
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
        let client_oa = Client::openai("sk-12345");
        assert_eq!(client_oa.provider(), &Provider::OpenAi);
        assert_eq!(client_oa.api_key(), Some("sk-12345"));

        let client_or = Client::openrouter("or-67890");
        assert_eq!(client_or.provider(), &Provider::openrouter());
        assert_eq!(client_or.api_key(), Some("or-67890"));

        let client_custom = Client::custom("http://localhost:11434/v1", None);
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
