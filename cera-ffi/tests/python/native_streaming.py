"""Run with a fresh binding module on PYTHONPATH and a local generative GGUF argument."""

import sys
import threading

import cera_ffi as c


def main(model_path):
    engine = c.CeraEngine.from_path(
        model_path, c.EngineConfig(context_size=256, backend=c.BackendPreference.CPU)
    )
    chat = engine.new_chat_session(c.SessionConfig(seed=42))
    chat.ingest(c.chat_message_user("Hi"))
    empty = c.GenerateOpts(max_tokens=0, temperature=0.0)
    for _ in range(100):
        assert list(chat.stream(empty)) == []
        assert not isinstance(chat.complete(empty).summary.finish_reason, c.FinishReason.CANCELLED)

    worker = []

    class ObservedSession:
        def generate_streaming(self, opts, sink):
            worker.append(threading.current_thread())
            return chat.generate_streaming(opts, sink)

        def cancel(self):
            chat.cancel()

    stream = c.chat_stream(ObservedSession(), c.GenerateOpts(max_tokens=1, temperature=0.0))
    next(stream)
    worker[0].join(30)
    assert not worker[0].is_alive(), "native generation did not finish"
    stream.close()
    chat.replace_messages([c.chat_message_user("again")])
    assert not isinstance(chat.complete(empty).summary.finish_reason, c.FinishReason.CANCELLED)
    print("PASS: 100 normal streams and closing a completed native worker preserve subsequent generation")


if __name__ == "__main__":
    main(sys.argv[1])
