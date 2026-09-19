#!/usr/bin/env python3
"""Multi-turn conversational chat with live KV cache retention in Python.

Demonstrates:
1. Loading a generative model and creating an execution session.
2. Converting the session into a transactional ChatSession.
3. Ingesting user turns and completing responses.
4. Retaining live KV context across consecutive turns without recomputation.
5. Reclaiming the underlying Session upon completion.

Run: `python3 cera-ffi/examples/chat.py model.gguf`
"""

import os
import sys
from pathlib import Path

# Add UniFFI Python bindings directory to module search path
BINDINGS_DIR = Path(__file__).resolve().parent.parent / "bindings" / "python"
sys.path.insert(0, str(BINDINGS_DIR))

try:
    import cera_ffi
except (ImportError, OSError) as err:
    print(f"Error loading cera_ffi Python module or shared library: {err}")
    print(
        "Build the matching cera-ffi shared library and place it beside cera_ffi.py "
        "in bindings/python (libcera_ffi.dylib, libcera_ffi.so, or cera_ffi.dll)."
    )
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("usage: chat.py <model.gguf>")
        sys.exit(1)

    model_path = sys.argv[1]
    print(f"Loading model from: {model_path}")

    loader = cera_ffi.ModelLoader(
        cera_ffi.ModelSource.PATH(model_path),
        cera_ffi.EngineConfig(backend=cera_ffi.BackendPreference.CPU),
    )
    model = loader.build_generative()
    session = model.create_session(cera_ffi.SessionConfig(seed=42))

    # Convert Session into transactional ChatSession.
    chat = session.into_chat()
    print(f"Initial chat phase: {chat.phase()}")
    print(f"Initial position: {chat.position()}")

    opts = cera_ffi.GenerateOpts(max_tokens=64, temperature=0.7)

    # --- Turn 1: Initialization with System and User messages ---
    print("\n--- Turn 1 ---")
    turn1_messages = [
        cera_ffi.chat_message_system("You are a helpful and concise systems engineering assistant."),
        cera_ffi.chat_message_user("What is a KV cache in LLM inference? Answer in one sentence."),
    ]

    summary1 = chat.ingest_messages(turn1_messages)
    print(
        f"Ingested {summary1.input_tokens} tokens "
        f"(position: {summary1.position_before} -> {summary1.position_after})"
    )
    print(f"Phase after ingest: {chat.phase()}")

    turn1 = chat.complete(opts)
    print(f"Assistant: {turn1.text.strip()}")
    print(f"Generated {turn1.summary.tokens_generated} tokens (final position: {chat.position()})")
    print(f"Phase after completion: {chat.phase()}")
    if chat.phase() != cera_ffi.SessionPhase.TURN_COMPLETE:
        print("Turn stopped before its terminal marker; reset or replace messages before a new user turn.")
        reclaimed_session = chat.into_session()
        print(f"Reclaimed raw session at position {reclaimed_session.position()}")
        return

    # --- Turn 2: Warm Continuation ---
    # The previous context remains in the KV cache; only new user input is ingested.
    print("\n--- Turn 2 (Warm Continuation) ---")
    turn2_user = cera_ffi.chat_message_user("When should it be discarded?")
    summary2 = chat.ingest(turn2_user)
    print(
        f"Ingested {summary2.input_tokens} new tokens "
        f"(position: {summary2.position_before} -> {summary2.position_after})"
    )

    turn2 = chat.complete(opts)
    print(f"Assistant: {turn2.text.strip()}")
    print(f"Generated {turn2.summary.tokens_generated} tokens (final position: {chat.position()})")
    print(f"Phase after completion: {chat.phase()}")
    # --- Reclaim raw Session ---
    # UniFFI Python bindings manage underlying Rust handles via reference counting and finalizers.
    reclaimed_session = chat.into_session()
    print(f"\nReclaimed raw session at position {reclaimed_session.position()}")


if __name__ == "__main__":
    main()
