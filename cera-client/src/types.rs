//! Strongly typed request and response definitions for OpenAI and OpenRouter endpoints.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Chat message author role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System prompt setting instructions and context.
    System,
    /// Developer prompt for reasoning models (e.g. o1/o3).
    Developer,
    /// User input message.
    User,
    /// Model assistant reply.
    Assistant,
    /// Result returned from an executed tool call.
    Tool,
    /// Result returned from a legacy function call.
    Function,
}

/// A message in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message author.
    pub role: Role,

    /// Text contents of the message. Optional when assistant returns tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Refusal message if the model refused to respond.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,

    /// Optional participant name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Tool calls made by the assistant, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// ID of the tool call this message is responding to (for `role = Role::Tool`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Construct a new system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(content.into()),
            refusal: None,
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Construct a new developer instruction message for reasoning models.
    pub fn developer(content: impl Into<String>) -> Self {
        Self {
            role: Role::Developer,
            content: Some(content.into()),
            refusal: None,
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Construct a new user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(content.into()),
            refusal: None,
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Construct a new assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content.into()),
            refusal: None,
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Construct a new assistant message containing tool calls.
    pub fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: None,
            refusal: None,
            name: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    /// Construct a new tool reply message for a given tool call ID.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            refusal: None,
            name: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    /// Construct a new legacy function reply message.
    pub fn function(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Function,
            content: Some(content.into()),
            refusal: None,
            name: Some(name.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

/// A tool call invoked by the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,

    /// Type of the tool call (usually `function`).
    #[serde(rename = "type")]
    pub call_type: String,

    /// Function call details including name and arguments string.
    pub function: FunctionCall,
}

/// Function invocation details in a tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Name of the function to call.
    pub name: String,

    /// JSON string representing the arguments passed to the function.
    pub arguments: String,
}

/// Specification of a tool the model can call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Type of the tool (currently `function`).
    #[serde(rename = "type")]
    pub tool_type: String,

    /// Function definition schema.
    pub function: FunctionDefinition,
}

impl ToolDefinition {
    /// Construct a function tool definition.
    pub fn function(
        name: impl Into<String>,
        description: Option<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.into(),
                description,
                parameters,
                strict: None,
            },
        }
    }

    /// Construct a function tool definition with strict schema adherence.
    pub fn strict_function(
        name: impl Into<String>,
        description: Option<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.into(),
                description,
                parameters,
                strict: Some(true),
            },
        }
    }
}

/// Schema definition for a callable function.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Function name.
    pub name: String,

    /// Description explaining what the function does and when to call it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON Schema describing the function parameter structure.
    pub parameters: serde_json::Value,

    /// Whether to enable strict schema adherence for structured outputs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl FunctionDefinition {
    /// Create a function definition without description or strict enforcement.
    pub fn new(name: impl Into<String>, parameters: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters,
            strict: None,
        }
    }

    /// Set function description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set strict schema mode.
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }
}

/// Request payload for `/chat/completions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model ID to query (e.g. `gpt-4o-mini`, `anthropic/claude-3.5-sonnet`).
    pub model: String,

    /// List of conversation messages.
    pub messages: Vec<ChatMessage>,

    /// Sampling temperature between 0.0 and 2.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Nucleus sampling probability cutoff (top-p).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Maximum number of tokens to generate in the completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Upper bound on completion tokens, used by reasoning models (e.g. o1/o3).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,

    /// Options for streaming responses, such as requesting token usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<serde_json::Value>,

    /// Whether to stream back partial progress via Server-Sent Events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// List of tools available to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,

    /// Tool choice policy (`none`, `auto`, `required`, or specific function).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,

    /// Output formatting requirement (e.g. `{"type": "json_object"}`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,

    /// Up to 4 sequences where the API will stop generating tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    /// Presence penalty between -2.0 and 2.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Frequency penalty between -2.0 and 2.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Random seed for deterministic generation if supported by provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    /// Optional end-user identifier for abuse detection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl ChatCompletionRequest {
    /// Create a new chat completion request with required model and messages.
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_completion_tokens: None,
            stream_options: None,
            stream: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            user: None,
        }
    }

    /// Set sampling temperature.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set nucleus sampling probability.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set maximum tokens to generate.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set maximum completion tokens for reasoning models (e.g. o1/o3).
    pub fn max_completion_tokens(mut self, max_completion_tokens: u32) -> Self {
        self.max_completion_tokens = Some(max_completion_tokens);
        self
    }

    /// Request token usage statistics in the final streaming chunk.
    pub fn include_usage(mut self) -> Self {
        self.stream_options = Some(serde_json::json!({ "include_usage": true }));
        self
    }

    /// Set streaming flag.
    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Attach tool definitions.
    pub fn tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set stop sequences.
    pub fn stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Set response format to JSON object (`{"type": "json_object"}`).
    pub fn json_mode(mut self) -> Self {
        self.response_format = Some(serde_json::json!({ "type": "json_object" }));
        self
    }
}

/// Token usage details for request and response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens in the prompt.
    pub prompt_tokens: u32,

    /// Tokens generated in the completion.
    pub completion_tokens: u32,

    /// Total tokens consumed.
    pub total_tokens: u32,
}

/// A choice generated by the model in a non-streaming completion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatChoice {
    /// Choice index.
    pub index: u32,

    /// Message generated by the model.
    pub message: ChatMessage,

    /// Reason generation stopped (e.g. `stop`, `length`, `tool_calls`).
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Response returned from `/chat/completions` for non-streaming requests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// Unique response identifier.
    pub id: String,

    /// Object type (typically `chat.completion`).
    #[serde(default)]
    pub object: Option<String>,

    /// Unix timestamp of completion creation.
    pub created: u64,

    /// Model that generated the completion.
    pub model: String,

    /// List of generated choices.
    pub choices: Vec<ChatChoice>,

    /// Token usage metrics, if reported by provider.
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// Streaming chunk delta for a message.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatChunkDelta {
    /// Author role if emitted on first chunk.
    #[serde(default)]
    pub role: Option<Role>,

    /// Generated text content increment.
    #[serde(default)]
    pub content: Option<String>,

    /// Incremental refusal message if model refused to answer.
    #[serde(default)]
    pub refusal: Option<String>,

    /// Incremental tool calls data.
    #[serde(default)]
    pub tool_calls: Option<Vec<ChunkToolCall>>,
}

/// Tool call delta within a streaming chunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkToolCall {
    /// Index of the tool call in the array.
    pub index: u32,

    /// Tool call ID, usually sent in the first chunk for this tool.
    #[serde(default)]
    pub id: Option<String>,

    /// Type string, usually `function`.
    #[serde(rename = "type", default)]
    pub call_type: Option<String>,

    /// Incremental function name and arguments fragments.
    #[serde(default)]
    pub function: Option<ChunkFunctionCall>,
}

/// Function call fragments in streaming chunks.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkFunctionCall {
    /// Name fragment, if emitted.
    #[serde(default)]
    pub name: Option<String>,

    /// Arguments JSON fragment.
    #[serde(default)]
    pub arguments: Option<String>,
}

/// A choice in a streaming chunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatChunkChoice {
    /// Choice index.
    pub index: u32,

    /// Incremental delta for this choice.
    pub delta: ChatChunkDelta,

    /// Reason generation stopped on the final chunk.
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Server-Sent Events chunk payload for streaming chat completions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    /// Unique response identifier.
    pub id: String,

    /// Object type (typically `chat.completion.chunk`).
    #[serde(default)]
    pub object: Option<String>,

    /// Unix timestamp of chunk creation.
    pub created: u64,

    /// Model name.
    pub model: String,

    /// Array of choice deltas.
    pub choices: Vec<ChatChunkChoice>,

    /// Token usage metrics if stream_options requested them.
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// Input text(s) for generating vector embeddings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddingInput {
    /// Single text prompt.
    Single(String),
    /// Batch of text prompts.
    Multiple(Vec<String>),
}

impl From<&str> for EmbeddingInput {
    fn from(s: &str) -> Self {
        Self::Single(s.to_string())
    }
}

impl From<String> for EmbeddingInput {
    fn from(s: String) -> Self {
        Self::Single(s)
    }
}

impl From<Vec<String>> for EmbeddingInput {
    fn from(v: Vec<String>) -> Self {
        Self::Multiple(v)
    }
}

impl Serialize for EmbeddingInput {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Single(text) => serializer.serialize_str(text),
            Self::Multiple(texts) => texts.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for EmbeddingInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Helper {
            Single(String),
            Multiple(Vec<String>),
        }
        match Helper::deserialize(deserializer)? {
            Helper::Single(s) => Ok(Self::Single(s)),
            Helper::Multiple(v) => Ok(Self::Multiple(v)),
        }
    }
}

/// Request payload for `/embeddings`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// Model ID (e.g. `text-embedding-3-small`).
    pub model: String,

    /// Input text or array of texts to embed.
    pub input: EmbeddingInput,

    /// Optional target dimensions for models supporting truncation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,

    /// Optional user ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl EmbeddingRequest {
    /// Construct a new embedding request.
    pub fn new(model: impl Into<String>, input: impl Into<EmbeddingInput>) -> Self {
        Self {
            model: model.into(),
            input: input.into(),
            dimensions: None,
            user: None,
        }
    }

    /// Set dimensions.
    pub fn dimensions(mut self, dims: u32) -> Self {
        self.dimensions = Some(dims);
        self
    }
}

/// Vector embedding for a single input text item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingData {
    /// Index of this embedding in the input list.
    pub index: u32,

    /// Object type (typically `embedding`).
    pub object: String,

    /// Float vector embedding.
    pub embedding: Vec<f32>,
}

/// Response returned from `/embeddings`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// Object type (`list`).
    pub object: String,

    /// List of embedding vectors.
    pub data: Vec<EmbeddingData>,

    /// Model used for embeddings.
    pub model: String,

    /// Token usage metrics.
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// Information describing a model available on the endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Unique model identifier (e.g. `gpt-4o`).
    pub id: String,

    /// Object type (typically `model`, optional on OpenRouter and Ollama).
    #[serde(default)]
    pub object: Option<String>,

    /// Unix timestamp when the model was created or added.
    #[serde(default)]
    pub created: Option<u64>,

    /// Organization or owner of the model.
    #[serde(default)]
    pub owned_by: Option<String>,
}

/// Response returned from `/models`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListModelsResponse {
    /// Object type (`list`, optional on some proxies).
    #[serde(default)]
    pub object: Option<String>,

    /// List of available models.
    pub data: Vec<ModelInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_constructors_and_serialization() {
        let msg = ChatMessage::system("System prompt");
        assert_eq!(msg.role, Role::System);
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"system\""));
        assert!(json.contains("\"content\":\"System prompt\""));

        let user_msg = ChatMessage::user("Hello");
        assert_eq!(user_msg.role, Role::User);

        let tool_msg = ChatMessage::tool("call_123", "{\"result\": 42}");
        assert_eq!(tool_msg.role, Role::Tool);
        assert_eq!(tool_msg.tool_call_id.as_deref(), Some("call_123"));
    }

    #[test]
    fn test_chat_completion_request_builder() {
        let tool = ToolDefinition::function(
            "get_weather",
            Some("Fetch weather for location".to_string()),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }),
        );

        let req = ChatCompletionRequest::new(
            "gpt-4o-mini",
            vec![ChatMessage::user("What is the weather?")],
        )
        .temperature(0.5)
        .top_p(0.9)
        .max_tokens(100)
        .tools(vec![tool])
        .json_mode()
        .stop(vec!["\n".to_string()]);

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "gpt-4o-mini");
        assert_eq!(json["temperature"], 0.5);
        assert_eq!(json["max_tokens"], 100);
        assert_eq!(json["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(json["response_format"]["type"], "json_object");
        assert_eq!(json["stop"][0], "\n");
    }

    #[test]
    fn test_chat_completion_response_deserialization() {
        let raw = r#"{
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        }"#;

        let res: ChatCompletionResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(res.id, "chatcmpl-123");
        assert_eq!(res.choices.len(), 1);
        assert_eq!(res.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(
            res.choices[0].message.content.as_deref(),
            Some("Hello there!")
        );
        assert_eq!(res.usage.unwrap().total_tokens, 21);
    }

    #[test]
    fn test_chat_completion_chunk_deserialization() {
        let raw = r#"{
            "id": "chatcmpl-chunk-1",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "part"
                },
                "finish_reason": null
            }]
        }"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.id, "chatcmpl-chunk-1");
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("part"));
        assert_eq!(chunk.choices[0].delta.role, Some(Role::Assistant));
    }

    #[test]
    fn test_embedding_request_and_response() {
        let req_single =
            EmbeddingRequest::new("text-embedding-3-small", "test text").dimensions(512);
        let val_single = serde_json::to_value(&req_single).unwrap();
        assert_eq!(val_single["input"], "test text");
        assert_eq!(val_single["dimensions"], 512);

        let req_multi = EmbeddingRequest::new(
            "text-embedding-3-small",
            vec!["item1".to_string(), "item2".to_string()],
        );
        let val_multi = serde_json::to_value(&req_multi).unwrap();
        assert_eq!(val_multi["input"][0], "item1");
        assert_eq!(val_multi["input"][1], "item2");

        let raw_res = r#"{
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, -0.2, 0.3]
                }
            ],
            "model": "text-embedding-3-small",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5,
                "completion_tokens": 0
            }
        }"#;
        let res: EmbeddingResponse = serde_json::from_str(raw_res).unwrap();
        assert_eq!(res.data.len(), 1);
        assert_eq!(res.data[0].embedding, vec![0.1, -0.2, 0.3]);
    }

    #[test]
    fn test_models_list_deserialization() {
        let raw = r#"{
            "object": "list",
            "data": [
                {
                    "id": "gpt-4o",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "openai"
                }
            ]
        }"#;
        let res: ListModelsResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(res.data.len(), 1);
        assert_eq!(res.data[0].id, "gpt-4o");
    }

    #[test]
    fn test_function_definition_and_role_serialization() {
        let msg = ChatMessage::function("calc", "42");
        let val = serde_json::to_value(&msg).unwrap();
        assert_eq!(val["role"], "function");
        assert_eq!(val["name"], "calc");
        assert_eq!(val["content"], "42");

        let tool_def = ToolDefinition::strict_function(
            "get_weather",
            Some("Fetch current weather".to_string()),
            serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"],
                "additionalProperties": false
            }),
        );
        let val_tool = serde_json::to_value(&tool_def).unwrap();
        assert_eq!(val_tool["type"], "function");
        assert_eq!(val_tool["function"]["name"], "get_weather");
        assert_eq!(val_tool["function"]["strict"], true);
    }
}
