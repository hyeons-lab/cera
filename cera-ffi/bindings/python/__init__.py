"""Python bindings for cera inference engine."""

import queue
import threading
from typing import Iterator

from .cera_ffi import (
    CeraEngine,
    ChatSession,
    EngineConfig,
    FfiEntitySpan,
    FinishReason,
    GenerateOpts,
    Message,
    ModalitySink,
    PiiClassifier,
    Role,
    Session,
    SessionPhase,
    TurnResult,
    ValidationError,
    chat_message_assistant,
    chat_message_system,
    chat_message_tool,
    chat_message_user,
    json_schema_to_grammar,
)


def chat_stream(session: ChatSession, opts: GenerateOpts) -> Iterator[str]:
    """Stream generated text fragments as a Python iterator.

    Exiting or breaking from the iterator loop triggers wait-free cancellation
    on the underlying ChatSession.
    """
    q: queue.Queue = queue.Queue()
    sentinel = object()

    class StreamSink(ModalitySink):
        def on_thought_chunk(self, text: str):
            pass

        def on_text_chunk(self, text: str):
            q.put(text)

        def on_audio_frames(self, pcm: list[float], sample_rate: int):
            pass

        def on_done(self, reason: FinishReason):
            if isinstance(reason, FinishReason.Error):
                q.put(RuntimeError(reason.message))
            else:
                q.put(sentinel)

    sink = StreamSink()

    def worker():
        try:
            session.generate_streaming(opts, sink)
        except Exception as exc:
            q.put(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        while True:
            item = q.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        session.cancel()


def chat_stream_json(session: ChatSession, opts: GenerateOpts, schema_json: str) -> Iterator[str]:
    """Stream generated text fragments constrained by a JSON Schema as a Python iterator."""
    grammar = json_schema_to_grammar(schema_json)
    opts.grammar = grammar
    return chat_stream(session, opts)


# Attach stream methods to ChatSession for idiomatic object-oriented calling
setattr(ChatSession, "stream", chat_stream)
setattr(ChatSession, "stream_json", chat_stream_json)

__all__ = [
    "CeraEngine",
    "ChatSession",
    "EngineConfig",
    "FfiEntitySpan",
    "FinishReason",
    "GenerateOpts",
    "Message",
    "ModalitySink",
    "PiiClassifier",
    "Role",
    "Session",
    "SessionPhase",
    "TurnResult",
    "ValidationError",
    "chat_message_assistant",
    "chat_message_system",
    "chat_message_tool",
    "chat_message_user",
    "chat_stream",
    "chat_stream_json",
    "json_schema_to_grammar",
]
