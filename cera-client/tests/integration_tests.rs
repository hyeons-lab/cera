use std::net::SocketAddr;
use std::time::Duration;

use cera_client::{ChatCompletionRequest, ChatMessage, Client, ClientError, EmbeddingRequest};
use futures_util::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

async fn read_request(socket: &mut tokio::net::TcpStream) -> String {
    let mut buf = vec![0u8; 4096];
    let mut bytes_read = 0;
    while bytes_read < buf.len() {
        let n = socket.read(&mut buf[bytes_read..]).await.unwrap();
        if n == 0 {
            break;
        }
        bytes_read += n;
        let s = String::from_utf8_lossy(&buf[..bytes_read]);
        if s.contains("\r\n\r\n") {
            return s.into_owned();
        }
    }
    String::from_utf8_lossy(&buf[..bytes_read]).into_owned()
}

/// Simple mock server to verify HTTP request formation and response handling.
async fn spawn_mock_server(
    expected_path: &'static str,
    response_headers: &'static str,
    response_body: &'static str,
) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request_str = read_request(&mut socket).await;

        assert!(
            request_str.contains(expected_path),
            "Request does not contain expected path: {request_str}"
        );

        let response = format!(
            "HTTP/1.1 200 OK\r\nConnection: close\r\n{response_headers}Content-Length: {}\r\n\r\n{response_body}",
            response_body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        socket.flush().await.unwrap();
        let _ = socket.shutdown().await;
    });

    (addr, handle)
}

#[tokio::test]
async fn test_mock_chat_completion() {
    let raw_res = r#"{
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello from mock server!"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }"#;

    let (addr, handle) = spawn_mock_server(
        "POST /chat/completions",
        "Content-Type: application/json\r\n",
        raw_res,
    )
    .await;

    let client = Client::custom(format!("http://{addr}"), Some("mock_key".to_string())).unwrap();
    let request = ChatCompletionRequest::new("gpt-4o", vec![ChatMessage::user("Hi")]);

    let response = client.chat(request).await.unwrap();
    assert_eq!(response.id, "chatcmpl-mock");
    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("Hello from mock server!")
    );
    assert_eq!(response.usage.unwrap().total_tokens, 15);

    handle.await.unwrap();
}

#[tokio::test]
async fn test_mock_chat_stream() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;

        let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
        socket.write_all(headers.as_bytes()).await.unwrap();

        let sse_data = [
            ": keep-alive\n\n",
            "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Chunk 1 \"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c2\",\"object\":\"chat.completion.chunk\",\"created\":2,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Chunk 2\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        ];

        for part in sse_data {
            let chunk = format!("{:X}\r\n{}\r\n", part.len(), part);
            socket.write_all(chunk.as_bytes()).await.unwrap();
            socket.flush().await.unwrap();
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        socket.write_all(b"0\r\n\r\n").await.unwrap();
        socket.flush().await.unwrap();
    });

    let client = Client::custom(format!("http://{addr}"), Some("mock_key".to_string())).unwrap();
    let request = ChatCompletionRequest::new("gpt-4o", vec![ChatMessage::user("Stream me")]);

    let mut stream = client.chat_stream(request).await.unwrap();
    let mut collected = String::new();

    while let Some(item) = stream.next().await {
        let chunk = item.unwrap();
        if let Some(choice) = chunk.choices.first()
            && let Some(content) = &choice.delta.content
        {
            collected.push_str(content);
        }
    }

    assert_eq!(collected, "Chunk 1 Chunk 2");
    handle.await.unwrap();
}

#[tokio::test]
async fn test_mock_api_error_rate_limit() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let error_body = r#"{
        "error": {
            "message": "Rate limit reached for requests",
            "type": "requests",
            "param": null,
            "code": "rate_limit_exceeded"
        }
    }"#;

    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;

        let response = format!(
            "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{error_body}",
            error_body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        socket.flush().await.unwrap();
    });

    let client = Client::custom(format!("http://{addr}"), Some("mock_key".to_string())).unwrap();
    let request = ChatCompletionRequest::new("gpt-4o", vec![ChatMessage::user("Hi")]);

    let err = client.chat(request).await.unwrap_err();
    assert!(err.is_rate_limited());
    match err {
        ClientError::Api {
            status,
            message,
            code,
            ..
        } => {
            assert_eq!(status, Some(reqwest::StatusCode::TOO_MANY_REQUESTS));
            assert!(message.contains("Rate limit reached"));
            assert_eq!(code.as_deref(), Some("rate_limit_exceeded"));
        }
        other => panic!("expected Api error, got: {other:?}"),
    }

    handle.await.unwrap();
}

#[tokio::test]
async fn test_mock_embeddings() {
    let raw_res = r#"{
        "object": "list",
        "data": [{
            "object": "embedding",
            "index": 0,
            "embedding": [0.05, -0.1, 0.25]
        }],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 4,
            "total_tokens": 4,
            "completion_tokens": 0
        }
    }"#;

    let (addr, handle) = spawn_mock_server(
        "POST /embeddings",
        "Content-Type: application/json\r\n",
        raw_res,
    )
    .await;

    let client = Client::custom(format!("http://{addr}"), Some("mock_key".to_string())).unwrap();
    let request = EmbeddingRequest::new("text-embedding-3-small", "hello world");

    let response = client.embeddings(request).await.unwrap();
    assert_eq!(response.data.len(), 1);
    assert_eq!(response.data[0].embedding, vec![0.05, -0.1, 0.25]);

    handle.await.unwrap();
}

#[tokio::test]
async fn test_mock_openrouter_numeric_error() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let error_body = r#"{"error":{"message":"Insufficient credits","code":402}}"#;

    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;

        let response = format!(
            "HTTP/1.1 402 Payment Required\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{error_body}",
            error_body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        socket.flush().await.unwrap();
    });

    let client = Client::custom(format!("http://{addr}"), Some("or-key".to_string())).unwrap();
    let request =
        ChatCompletionRequest::new("anthropic/claude-3.5-sonnet", vec![ChatMessage::user("Hi")]);

    let err = client.chat(request).await.unwrap_err();
    match err {
        ClientError::Api {
            status,
            message,
            code,
            ..
        } => {
            assert_eq!(status, Some(reqwest::StatusCode::PAYMENT_REQUIRED));
            assert_eq!(message, "Insufficient credits");
            assert_eq!(code.as_deref(), Some("402"));
        }
        other => panic!("expected Api error, got: {other:?}"),
    }

    handle.await.unwrap();
}

#[tokio::test]
async fn test_mock_models_without_object() {
    let raw_models = r#"{
        "data": [{
            "id": "openrouter/auto",
            "created": 1700000000,
            "owned_by": "openrouter"
        }]
    }"#;

    let (addr, handle) = spawn_mock_server(
        "GET /models",
        "Content-Type: application/json\r\n",
        raw_models,
    )
    .await;

    let client = Client::custom(format!("http://{addr}"), None).unwrap();
    let response = client.models().await.unwrap();
    assert_eq!(response.object, None);
    assert_eq!(response.data.len(), 1);
    assert_eq!(response.data[0].id, "openrouter/auto");
    assert_eq!(response.data[0].object, None);

    handle.await.unwrap();
}
