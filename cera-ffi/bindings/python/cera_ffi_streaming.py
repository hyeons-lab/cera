"""Streaming helpers installed on the generated Python module by the binding recipe."""

import copy
import queue
import threading
from typing import Iterator


def install(bindings):
    ChatSession = bindings.ChatSession
    GenerateOpts = bindings.GenerateOpts
    FinishReason = bindings.FinishReason
    ModalitySink = bindings.ModalitySink
    json_schema_to_grammar = bindings.json_schema_to_grammar

    def chat_stream(session: ChatSession, opts: GenerateOpts) -> Iterator[str]:
        """Stream generated text fragments as a Python iterator.

        Closing this iterator requests cancellation if generation is still active.
        """
        q: queue.Queue = queue.Queue()
        sentinel = object()
        terminal_error = None
        worker_done = threading.Event()

        class StreamSink(ModalitySink):
            def on_thought_chunk(self, text: str):
                pass

            def on_text_chunk(self, text: str):
                q.put(text)

            def on_audio_frames(self, pcm: list[float], sample_rate: int):
                pass

            def on_done(self, reason: FinishReason):
                nonlocal terminal_error
                if isinstance(reason, FinishReason.ERROR):
                    terminal_error = RuntimeError(reason.message)

        sink = StreamSink()

        def worker():
            outcome = sentinel
            try:
                session.generate_streaming(opts, sink)
            except Exception as exc:
                outcome = exc
            finally:
                # on_done runs before the native call releases its chat lock.
                # Expose completion only once a following operation can begin.
                worker_done.set()
                q.put(terminal_error if terminal_error is not None else outcome)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        finished = False
        try:
            while True:
                item = q.get()
                if item is sentinel:
                    finished = True
                    break
                if isinstance(item, Exception):
                    finished = True
                    raise item
                yield item
        finally:
            if not finished and not worker_done.is_set():
                session.cancel()


    def chat_stream_json(session: ChatSession, opts: GenerateOpts, schema_json: str) -> Iterator[str]:
        """Stream generated text fragments constrained by a JSON Schema as a Python iterator."""
        constrained_opts = copy.copy(opts)
        grammar = json_schema_to_grammar(schema_json)
        constrained_opts.grammar = grammar
        return chat_stream(session, constrained_opts)


    def with_json_schema(opts: GenerateOpts, schema_json: str) -> GenerateOpts:
        """Return a copy of GenerateOpts constrained by the provided JSON Schema."""
        new_opts = copy.copy(opts)
        new_opts.grammar = json_schema_to_grammar(schema_json)
        return new_opts


    bindings.chat_stream = chat_stream
    bindings.chat_stream_json = chat_stream_json
    bindings.with_json_schema = with_json_schema
    ChatSession.stream = chat_stream
    ChatSession.stream_json = chat_stream_json
    GenerateOpts.with_json_schema = with_json_schema
