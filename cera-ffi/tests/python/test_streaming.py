"""Run with the generated module, helper, and fresh native library on PYTHONPATH."""

import threading
import unittest

import cera_ffi as c


class StreamingTests(unittest.TestCase):
    def test_documented_entry_point_installs_helpers(self):
        self.assertIs(c.ChatSession.stream, c.chat_stream)
        self.assertIs(c.ChatSession.stream_json, c.chat_stream_json)
        self.assertIs(c.GenerateOpts.with_json_schema, c.with_json_schema)
        original = c.GenerateOpts()
        constrained = original.with_json_schema('{"type":"integer"}')
        self.assertIsNone(original.grammar)
        self.assertIn("json-integer", constrained.grammar)

    def test_normal_and_error_terminal_variants(self):
        class Session:
            def __init__(self, reason):
                self.reason = reason
                self.cancelled = False

            def generate_streaming(self, opts, sink):
                sink.on_text_chunk("é")
                sink.on_done(self.reason)

            def cancel(self):
                self.cancelled = True

        session = Session(c.FinishReason.MAX_TOKENS())
        self.assertEqual(list(c.chat_stream(session, c.GenerateOpts())), ["é"])
        self.assertFalse(session.cancelled)
        session = Session(c.FinishReason.ERROR(message="injected"))
        with self.assertRaisesRegex(RuntimeError, "injected"):
            list(c.chat_stream(session, c.GenerateOpts()))

    def test_closing_iterator_cancels_active_generation(self):
        cancelled = threading.Event()
        exited = threading.Event()

        class Session:
            def generate_streaming(self, opts, sink):
                sink.on_text_chunk("first")
                cancelled.wait(2)
                sink.on_done(c.FinishReason.CANCELLED())
                exited.set()

            def cancel(self):
                cancelled.set()

        stream = c.chat_stream(Session(), c.GenerateOpts())
        self.assertEqual(next(stream), "first")
        stream.close()
        self.assertTrue(cancelled.is_set())
        self.assertTrue(exited.wait(2))

    def test_terminal_callback_does_not_finish_iterator_before_native_return(self):
        terminal_called = threading.Event()
        release_native = threading.Event()
        consumed = threading.Event()

        class Session:
            def generate_streaming(self, opts, sink):
                sink.on_done(c.FinishReason.MAX_TOKENS())
                terminal_called.set()
                release_native.wait(2)

            def cancel(self):
                release_native.set()

        def consume():
            list(c.chat_stream(Session(), c.GenerateOpts()))
            consumed.set()

        consumer = threading.Thread(target=consume)
        consumer.start()
        try:
            self.assertTrue(terminal_called.wait(2))
            self.assertFalse(consumed.wait(0.05))
        finally:
            release_native.set()
            consumer.join(2)
        self.assertTrue(consumed.is_set())

    def test_closing_after_worker_return_does_not_cancel(self):
        cancelled = threading.Event()
        worker = []

        class Session:
            def generate_streaming(self, opts, sink):
                worker.append(threading.current_thread())
                sink.on_text_chunk("last")
                sink.on_done(c.FinishReason.MAX_TOKENS())

            def cancel(self):
                cancelled.set()

        stream = c.chat_stream(Session(), c.GenerateOpts())
        self.assertEqual(next(stream), "last")
        worker[0].join(2)
        self.assertFalse(worker[0].is_alive())
        stream.close()
        self.assertFalse(cancelled.is_set())


if __name__ == "__main__":
    unittest.main()
