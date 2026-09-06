//! Asynchronous client library for OpenAI and OpenRouter endpoints.
//!
//! Provides strongly typed request and response structures, real-time Server-Sent Events
//! (SSE) stream parsing, tool calling schemas, and provider abstractions.

pub mod client;
pub mod error;
pub mod provider;
pub mod stream;
pub mod types;

pub use client::{Client, ClientBuilder};
pub use error::{ApiErrorEnvelope, ApiErrorPayload, ClientError};
pub use provider::{
    OPENAI_API_KEY_ENV, OPENAI_BASE_URL_ENV, OPENAI_DEFAULT_BASE_URL, OPENROUTER_API_KEY_ENV,
    OPENROUTER_DEFAULT_BASE_URL, Provider,
};
pub use reqwest::StatusCode;
pub use stream::{BoxChatCompletionStream, ChatCompletionStream};
pub use types::{
    ChatChoice, ChatChunkChoice, ChatChunkDelta, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage, ChunkFunctionCall, ChunkToolCall, EmbeddingData,
    EmbeddingInput, EmbeddingRequest, EmbeddingResponse, FunctionCall, FunctionDefinition,
    ListModelsResponse, ModelInfo, Role, ToolCall, ToolDefinition, Usage,
};
