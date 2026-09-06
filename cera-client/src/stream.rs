//! Server-Sent Events (SSE) stream decoder for chat completion chunks.

use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::{Bytes, BytesMut};
use futures_core::stream::{FusedStream, Stream};

use crate::error::{ApiErrorEnvelope, ClientError};
use crate::types::ChatCompletionChunk;

/// A pinned, heap-allocated stream yielding incremental [`ChatCompletionChunk`] updates.
pub type BoxChatCompletionStream =
    ChatCompletionStream<Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>>;

/// Asynchronous stream yielding incremental [`ChatCompletionChunk`] updates from an SSE stream.
pub struct ChatCompletionStream<S> {
    inner: S,
    buffer: BytesMut,
    done: bool,
}

impl<S> std::fmt::Debug for ChatCompletionStream<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatCompletionStream")
            .field("buffered_bytes", &self.buffer.len())
            .field("done", &self.done)
            .finish()
    }
}

impl<S> ChatCompletionStream<S> {
    /// Create a new stream wrapper around an inner byte stream.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: BytesMut::new(),
            done: false,
        }
    }
}

impl<S> ChatCompletionStream<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    /// Pin and box this stream into a [`BoxChatCompletionStream`].
    pub fn boxed(self) -> BoxChatCompletionStream {
        ChatCompletionStream {
            inner: Box::pin(self.inner),
            buffer: self.buffer,
            done: self.done,
        }
    }
}

impl<S> ChatCompletionStream<S> {
    /// Helper to process a single decoded line.
    ///
    /// Returns:
    /// - `Some(Ok(chunk))` if a valid chunk was parsed
    /// - `Some(Err(err))` if an error or in-band API error occurred
    /// - `None` if the line was empty, a comment, other SSE field, or `[DONE]`
    fn process_line(line: &str) -> (Option<Result<ChatCompletionChunk, ClientError>>, bool) {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(':') {
            // Keep-alive or comment line, ignore.
            return (None, false);
        }

        if let Some(payload) = trimmed.strip_prefix("data:") {
            let data = payload.trim();
            if data.is_empty() {
                // Keep-alive empty data heartbeat, ignore.
                return (None, false);
            }
            if data == "[DONE]" {
                return (None, true);
            }

            match serde_json::from_str::<ChatCompletionChunk>(data) {
                Ok(chunk) => (Some(Ok(chunk)), false),
                Err(err) => {
                    if let Ok(env) = serde_json::from_str::<ApiErrorEnvelope>(data) {
                        (
                            Some(Err(ClientError::Api {
                                status: None,
                                message: env.error.message,
                                error_type: env.error.error_type,
                                code: env.error.code,
                                param: env.error.param,
                            })),
                            true,
                        )
                    } else {
                        (Some(Err(ClientError::Serialization(err))), false)
                    }
                }
            }
        } else {
            (None, false)
        }
    }
}

/// Maximum allowed buffer capacity for an SSE line (16 MB) to guard against unbounded streams.
const MAX_STREAM_BUFFER_BYTES: usize = 16 * 1024 * 1024;

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
            if let Some(idx) = this.buffer.iter().position(|&b| b == b'\n') {
                let line_bytes = this.buffer.split_to(idx + 1);
                let mut slice = line_bytes.as_ref();
                if slice.ends_with(b"\n") {
                    slice = &slice[..slice.len() - 1];
                }
                if slice.ends_with(b"\r") {
                    slice = &slice[..slice.len() - 1];
                }

                let line = match std::str::from_utf8(slice) {
                    Ok(s) => s,
                    Err(e) => {
                        this.done = true;
                        return Poll::Ready(Some(Err(ClientError::Stream(format!(
                            "Invalid UTF-8 in SSE stream line: {e}"
                        )))));
                    }
                };

                let (result, is_done) = Self::process_line(line);
                if is_done {
                    this.done = true;
                }
                if let Some(res) = result {
                    return Poll::Ready(Some(res));
                }
                if is_done {
                    return Poll::Ready(None);
                }
                // If it was a comment or empty line, continue processing buffer
                continue;
            }

            // Need more data from the network
            match Pin::new(&mut this.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    if this.buffer.len() + bytes.len() > MAX_STREAM_BUFFER_BYTES {
                        this.done = true;
                        return Poll::Ready(Some(Err(ClientError::Stream(
                            "SSE stream line exceeded maximum buffer capacity of 16 MB".to_string(),
                        ))));
                    }
                    this.buffer.extend_from_slice(&bytes);
                }
                Poll::Ready(Some(Err(e))) => {
                    this.done = true;
                    return Poll::Ready(Some(Err(ClientError::Http(e))));
                }
                Poll::Ready(None) => {
                    // Stream closed
                    if !this.buffer.is_empty() {
                        let remaining = std::mem::take(&mut this.buffer);
                        let mut slice = remaining.as_ref();
                        if slice.ends_with(b"\n") {
                            slice = &slice[..slice.len() - 1];
                        }
                        if slice.ends_with(b"\r") {
                            slice = &slice[..slice.len() - 1];
                        }

                        if !slice.is_empty() {
                            let line = match std::str::from_utf8(slice) {
                                Ok(s) => s,
                                Err(e) => {
                                    this.done = true;
                                    return Poll::Ready(Some(Err(ClientError::Stream(format!(
                                        "Invalid UTF-8 in trailing SSE data: {e}"
                                    )))));
                                }
                            };

                            let (result, is_done) = Self::process_line(line);
                            this.done = true;
                            if let Some(res) = result {
                                return Poll::Ready(Some(res));
                            }
                            if is_done {
                                return Poll::Ready(None);
                            }
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

impl<S> FusedStream for ChatCompletionStream<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    fn is_terminated(&self) -> bool {
        self.done
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
    async fn test_sse_stream_handles_multibyte_utf8_split_across_chunks() {
        // Japanese character "あ" is [0xE3, 0x81, 0x82] in UTF-8
        let prefix = "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"";
        let suffix = "\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n";

        let mut chunk1_bytes = prefix.as_bytes().to_vec();
        chunk1_bytes.push(0xE3);
        chunk1_bytes.push(0x81); // Split before last byte

        let mut chunk2_bytes = vec![0x82];
        chunk2_bytes.extend_from_slice(suffix.as_bytes());

        let byte_stream = futures_util::stream::iter(vec![
            Ok::<Bytes, reqwest::Error>(Bytes::from(chunk1_bytes)),
            Ok::<Bytes, reqwest::Error>(Bytes::from(chunk2_bytes)),
        ]);

        let mut sse_stream = ChatCompletionStream::new(byte_stream);
        let item = sse_stream.next().await.expect("item 1").unwrap();
        assert_eq!(item.choices[0].delta.content.as_deref(), Some("あ"));
        assert!(sse_stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_sse_stream_handles_midstream_api_error() {
        let payload = "data: {\"error\":{\"message\":\"Model overloaded\",\"type\":\"server_error\",\"code\":\"server_error\"}}\n\n";
        let byte_stream =
            futures_util::stream::iter(vec![Ok::<Bytes, reqwest::Error>(Bytes::from(payload))]);
        let mut sse_stream = ChatCompletionStream::new(byte_stream);

        let err = sse_stream.next().await.expect("item").unwrap_err();
        match err {
            ClientError::Api { message, code, .. } => {
                assert_eq!(message, "Model overloaded");
                assert_eq!(code.as_deref(), Some("server_error"));
            }
            other => panic!("expected Api error, got: {other:?}"),
        }
        assert!(sse_stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_sse_stream_handles_invalid_utf8() {
        let bad_bytes = Bytes::from(vec![0xff, 0xfe, 0xfd, b'\n']);
        let byte_stream = futures_util::stream::iter(vec![Ok::<Bytes, reqwest::Error>(bad_bytes)]);
        let mut sse_stream = ChatCompletionStream::new(byte_stream);

        let err = sse_stream.next().await.expect("error").unwrap_err();
        match err {
            ClientError::Stream(msg) => assert!(msg.contains("Invalid UTF-8")),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_sse_stream_handles_empty_data_heartbeat() {
        let stream_text = "data: \n\ndata: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata:\n\ndata: [DONE]\n\n";
        let byte_stream =
            futures_util::stream::iter(vec![Ok::<Bytes, reqwest::Error>(Bytes::from(stream_text))]);
        let mut sse_stream = ChatCompletionStream::new(byte_stream);

        let item = sse_stream.next().await.expect("item").unwrap();
        assert_eq!(item.choices[0].delta.content.as_deref(), Some("ok"));
        assert!(sse_stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_fused_stream_and_boxed() {
        let stream_text = "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"test\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        let byte_stream =
            futures_util::stream::iter(vec![Ok::<Bytes, reqwest::Error>(Bytes::from(stream_text))]);
        let mut boxed_stream = ChatCompletionStream::new(byte_stream).boxed();

        assert!(!boxed_stream.is_terminated());
        let item = boxed_stream.next().await.expect("item").unwrap();
        assert_eq!(item.choices[0].delta.content.as_deref(), Some("test"));
        assert!(boxed_stream.next().await.is_none());
        assert!(boxed_stream.is_terminated());
        // FusedStream guarantees subsequent polls return None without panicking
        assert!(boxed_stream.next().await.is_none());
    }
}
