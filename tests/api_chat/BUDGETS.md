# Warm-Session Performance & Memory Regression Budgets

This specification documents the frozen performance and memory regression budgets required by **Section 8.1 (Warm-session performance gates)** of `API_RESHAPE_PLAN.md` for Phase 1 (P1 Delivery) and Phase 2 (P2 Migration).

## 1. Deterministic Work Invariants

During normal multi-turn conversational chat within context capacity:

1. **Delta-Only Prompt Evaluation:** Each conversational turn must evaluate only the newly supplied user/tool message tokens plus the profile's pending continuation prefix. Previously committed context must remain resident in the KV cache without re-evaluation.
2. **Zero History Replay:** A successful turn must never re-parse, re-tokenize, or re-forward previously committed conversational history.
3. **Zero Normal-Turn Resets:** In-capacity turns must never invoke device or session reset. Reset is reserved exclusively for deliberate history truncation, manual replacement, or recovery after fatal fault.
4. **Zero Full-Cache Copies:** Normal turns must not copy, snapshot, or read back the KV cache to host memory for checkpointing or convenience.
5. **Bit-Exact KV Retention:** All key and value tensors for prior token positions must match with bitwise equality (`diff == 0.0`) across consecutive turns.

## 2. Quantitative Targets & Regression Budgets

Budgets defined in [`budgets.json`](budgets.json) establish strict gates against bare raw `Session::append_tokens` / `Session::generate` references:

### A. Turn Framing & Ingestion Latency
* **Metric:** Time elapsed from `Chat::ingest` / `ChatSession::ingest` entry until prompt tokens are committed and execution phase advances to `PromptReady`.
* **Budget:** Wrapper and template boundary framing overhead must not exceed **1.5 ms** on reference Apple Silicon / ARM64 hardware.

### B. Time to First Visible Output (TTFT)
* **Metric:** Paired ratio of end-to-end time to first emitted token on warm turn $N$ versus a matched retained-session raw prefill (`Session::append_tokens` / `Session::generate`) of identical delta token length with matched resident context length.
* **Budget:** $\text{TTFT}_{\text{warm}} \le 1.05 \times \text{TTFT}_{\text{raw}}$ (maximum 5% overhead attributable to session coordinator book-keeping).

### C. Decode Throughput
* **Metric:** Generated tokens per second during `Chat::complete` / `ChatSession::complete` compared directly against bare raw `Session::generate`.
* **Budget:** $\text{Throughput}_{\text{chat}} \ge 0.98 \times \text{Throughput}_{\text{raw}}$ (at least 98% of raw session throughput).

### D. Memory Overhead
* **Coordinator Struct Overhead:** Chat coordinator heap allocation outside the underlying `Session` must not exceed **16 KiB** total.
* **Per-Turn Allocation Volume:** Temporary per-turn string allocations and token buffers must not exceed **8 KiB** per completed turn.

## 3. Hardware Reference Targets & Roadmap

Budgets in [`budgets.json`](budgets.json) target the `native-cpu-arm64` reference profile for Phase 1 delivery. Subsequent workstreams in Phase 2 will establish measured baselines for `native-cpu-x86_64` and GPU/Metal acceleration backends before activating strict budget enforcement on those architectures.
