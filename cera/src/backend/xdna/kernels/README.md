# XDNA kernels

IRON (mlir-aie) designs for the AMD XDNA backend, with their compiled
artifacts committed next to the sources. Two kernels cover a
Llama-3.2-1B decode token under one hardware context each:

- `q4layer` — a full transformer layer (RMSNorms, seven Q4 GEMV stages,
  interleaved-pair RoPE on `q`/`k`, single-query GQA online-softmax
  attention, `SiLU(gate)*up`, and residual adds) in a single run (the
  fused design from `docs/XDNA_PHASE0_RESULTS.md` section 6; per-shape
  xclbins are not used because switching costs ~800 us).
- `q4head` — the 131072x2048 GEMV for the LM head (vocab padded from
  128256), the `q4gemv` design at head shape.

Sources are in `iron/` (cera's own code, Apache-2.0 OR MIT, developed in
the Phase 0b spike under `tools/xdna-spike/`). Regenerate with
`build_kernels.sh` (WSL2, pinned mlir-aie v1.4.3 + Peano
22.0.0.2026090701); it prints nothing to update by hand — refresh the
hashes below in the same commit as any artifact change. Instruction binaries are deterministic: their hashes must match this table after a rebuild. The `.xclbin` files embed build metadata and are not bit-deterministic, so their hashes are provenance for the committed files only; the drift check for them is a rebuild plus a passing `spike q4layer` / `xdna_probe` run.

| Artifact | SHA-256 |
|---|---|
| `q4layer.xclbin` | `39a7b25aaa9fcbbe4ed3391a9811be92f225d1aa63bb339bc7c49302cca5acf4` |
| `insts_q4layer.bin` | `24247f60dd5e3c1379122189e6dee8824780bd3450af431bb75ce06bb3f918d7` |
| `q4head.xclbin` | `39a7b25aaa9fcbbe4ed3391a9811be92f225d1aa63bb339bc7c49302cca5acf4` |
| `insts_q4head.bin` | `5f49c3a90f429c93cd79d27021d8ce1388e336a0cfd90fb875c0baf6186ec6d6` |

Weight and activation formats are the decision record's
(`docs/XDNA_PHASE0_RESULTS.md` section 4): signed int4 weights (Q4_0's
`q - 8`), one bf16 scale per 32-weight block per row, blocks-then-scales
tile layout, Q8_0-style int8 activations quantized on the core.
