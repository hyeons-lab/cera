# cera-client

Asynchronous, strongly typed Rust client for OpenAI and OpenRouter endpoints.

Part of the [Cera](https://github.com/hyeons-lab/cera) inference workspace.

## Features

- **Multi-Provider Support**: First-class support for OpenAI, OpenRouter, and custom OpenAI-compatible servers (vLLM, Ollama, LocalAI, etc.).
- **Chat Completions**: Typed request and response models with tool calling, JSON schema output, and custom sampling parameters.
- **Streaming Responses**: Real-time Server-Sent Events (SSE) stream parsing yielding chunk deltas.
- **Embeddings**: Generate vector embeddings for text with batch input support.
- **Model Discovery**: Fetch available model catalogs from providers.
- **Environment Auto-Discovery**: Automatically resolve `OPENAI_API_KEY` or `OPENROUTER_API_KEY` when available.

## Usage

```rust,no_run
use cera_client::{ChatMessage, ChatCompletionRequest, Client};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connects to OpenRouter or OpenAI using environment variables
    let client = Client::from_env()?;

    let request = ChatCompletionRequest::new(
        "gpt-4o-mini",
        vec![
            ChatMessage::system("You are a concise assistant."),
            ChatMessage::user("Explain Rust ownership in two sentences."),
        ],
    )
    .temperature(0.7)
    .max_tokens(150);

    let response = client.chat(request).await?;
    if let Some(choice) = response.choices.first() {
        if let Some(content) = &choice.message.content {
            println!("{content}");
        }
    }

    Ok(())
}
```
