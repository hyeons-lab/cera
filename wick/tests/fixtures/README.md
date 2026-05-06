# Test fixtures

## `pug.jpg` (optional)

End-to-end VL smoke test (`vl_bundle_appends_synthetic_image` in
`tests/vl_bundle_load.rs`) prefers this fixture when present —
small JPEG of a recognisable subject so manual output reads
nicely. When absent the test falls back to a synthesised solid-
red 256² PNG. Either input lands in the same assertion shape
(LLM produces non-degenerate text); the fixture only changes
how interesting the manual output looks.

Not committed to keep the repo small. Drop a small JPEG in this
directory and the gated test (`WICK_TEST_DOWNLOAD=1 cargo test
--test vl_bundle_load -- --ignored`) will pick it up automatically.
