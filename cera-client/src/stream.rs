//! Server-Sent Events (SSE) stream decoder for chat completion chunks.

use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_core::Stream;

use crate::error::ClientError;
use crate::types::ChatCompletionChunk;

/// Asynchronous stream yielding incremental [`ChatCompletionChunk`] updates from an SSE stream.
pub struct ChatCompletionStream<S> {
    inner: S,
    buffer: String,
    done: bool,
}

impl<S> ChatCompletionStream<S> {
    /// Create a new stream wrapper around an inner byte stream.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: String::new(),
            done: false,
        }
    }

    /// Helper to extract and process a single line from the internal buffer.
    fn process_line(line: &str) -> Option<Result<ChatCompletionChunk, ClientError>> {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(':') {
            // Keep-alive or comment line, ignore.
            return None;
        }

        if let Some(payload) = trimmed.strip_prefix("data:") {
            let data = payload.trim();
            if data == "[DONE]" {
                return None;
            }

            match serde_json::from_str::<ChatCompletionChunk>(data) {
                Ok(chunk) => Some(Ok(chunk)),
                Err(err) => Some(Err(ClientError::Serialization(err))),
            }
        } else {
            None
        }
    }
}

impl<S> Stream for ChatCompletionStream<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<ChatCompletionChunk, ClientError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.as_mut().get_mut();

        loop {
            if this.done {
                return Poll::Ready(None);
            }

            // Check if we have a full line in the buffer
            if let Some(idx) = this.buffer.find('\n') {
                let mut line = this.buffer.drain(..=idx).collect::<String>();
                if line.ends_with('\n') {
                    line.pop();
                    if line.ends_with('\r') {
                        line.pop();
                    }
                }

                let trimmed = line.trim();
                if trimmed == "data: [DONE]" || trimmed == "data:[DONE]" {
                    this.done = true;
                    return Poll::Ready(None);
                }

                if let Some(result) = Self::process_line(&line) {
                    return Poll::Ready(Some(result));
                }
                // If it was a comment or empty line, continue processing buffer
                continue;
            }

            // Need more data from the network
            match Pin::new(&mut this.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => match std::str::from_utf8(&bytes) {
                    Ok(text) => {
                        this.buffer.push_str(text);
                    }
                    Err(e) => {
                        this.done = true;
                        return Poll::Ready(Some(Err(ClientError::Stream(format!(
                            "Invalid UTF-8 in SSE stream: {e}"
                        )))));
                    }
                },
                Poll::Ready(Some(Err(e))) => {
                    this.done = true;
                    return Poll::Ready(Some(Err(ClientError::Http(e))));
                }
                Poll::Ready(None) => {
                    // Stream closed
                    if !this.buffer.is_empty() {
                        let remaining = std::mem::take(&mut this.buffer);
                        let trimmed = remaining.trim();
                        if trimmed == "data: [DONE]" || trimmed == "data:[DONE]" {
                            this.done = true;
                            return Poll::Ready(None);
                        }
                        if let Some(result) = Self::process_line(&remaining) {
                            this.done = true;
                            return Poll::Ready(Some(result));
                        }
                    }
                    this.done = true;
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    #[tokio::test]
    async fn test_sse_stream_parsing_with_fragmented_chunks() {
        let chunk1 = Bytes::from(
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":123,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel",
        );
        let chunk2 = Bytes::from(
            "lo\"},\"finish_reason\":null}]}\n\n: keep-alive\n\ndata: {\"id\":\"2\",\"object\":\"chat.completion.chunk\",\"created\":124,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
        );

        let byte_stream = futures_util::stream::iter(vec![
            Ok::<Bytes, reqwest::Error>(chunk1),
            Ok::<Bytes, reqwest::Error>(chunk2),
        ]);

        let mut sse_stream = ChatCompletionStream::new(byte_stream);

        let item1 = sse_stream.next().await.expect("item 1").unwrap();
        assert_eq!(item1.id, "1");
        assert_eq!(item1.choices[0].delta.content.as_deref(), Some("Hello"));

        let item2 = sse_stream.next().await.expect("item 2").unwrap();
        assert_eq!(item2.id, "2");
        assert_eq!(item2.choices[0].delta.content.as_deref(), Some(" world"));
        assert_eq!(item2.choices[0].finish_reason.as_deref(), Some("stop"));

        assert!(sse_stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_sse_stream_handles_invalid_utf8() {
        let bad_bytes = Bytes::from(vec![0xff, 0xfe, 0xfd]);
        let byte_stream = futures_util::stream::iter(vec![Ok::<Bytes, reqwest::Error>(bad_bytes)]);
        let mut sse_stream = ChatCompletionStream::new(byte_stream);

        let err = sse_stream.next().await.expect("error").unwrap_err();
        match err {
            ClientError::Stream(msg) => assert!(msg.contains("Invalid UTF-8")),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
