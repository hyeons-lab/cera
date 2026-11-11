# XDNA kernels

IRON (mlir-aie) designs for the AMD XDNA backend, with their compiled
artifacts committed next to the sources. Two kernels cover a
Llama-3.2-1B decode token under one shared hardware context:

- `q4layer` — a full transformer layer (RMSNorms, seven Q4 GEMV stages
  packed into four per-worker weight groups `(q,k,v)`, `o`, `(gate,up)`,
  `down`, interleaved-pair RoPE on `q`/`k` from L1-cached cos/sin,
  16-token block online-softmax GQA attention, on-core `SiLU(gate)*up`
  fusion, and L1-cached residual adds) in a single run. Nine instruction
  streams (`insts_q4layer_c1..c8.bin` and `insts_q4layer.bin` for `c9`,
  72 cached tokens per chunk) share `q4layer.xclbin` so only populated
  KV chunks are streamed without switching hardware contexts.
- `q4head` — the 131072x2048 GEMV for the LM head (vocab padded from
  128256), sharing `q4layer.xclbin` (`Q4LAYER_HEAD=1`).

Sources are in `iron/` (cera's own code, Apache-2.0 OR MIT, developed in
the Phase 0b spike under `tools/xdna-spike/`). Regenerate with
`build_kernels.sh` (WSL2, pinned mlir-aie v1.4.3 + Peano
22.0.0.2026090701); it prints nothing to update by hand — refresh the
hashes below in the same commit as any artifact change. Instruction binaries are deterministic: their hashes must match this table after a rebuild. The `.xclbin` files embed build metadata and are not bit-deterministic, so their hashes are provenance for the committed files only; the drift check for them is a rebuild plus a passing `xdna_layer_probe` / `xdna_probe` run.

| Artifact | SHA-256 |
|---|---|
| `q4layer.xclbin` | `eb4967aeb57fb5d713acc7a272770e4ac03e13566be866e84450b6ef93d2613d` |
| `insts_q4layer.bin` | `06c54fd42e94a8ae2a9b9e4145a417f86e1419b89f7f5e38acc6ea8505f9c25e` |
| `insts_q4layer_c1.bin` | `16d3f98e052179ddefdf8efdd7900acef4ea1d655dc5a2a59ad013b9918588fc` |
| `insts_q4layer_c2.bin` | `7afe7b7e915b9ca181af1a8bd551efb86fbe3d3850a8e8e8faef158046699af1` |
| `insts_q4layer_c3.bin` | `789affd671df65e857e88c41ed6d78843f695837bff30a402bd9f872bde5e6bb` |
| `insts_q4layer_c4.bin` | `6c1b7110c556355525b3d0186b5877daf7fbf8e1c922e5247fa483d1325a89bd` |
| `insts_q4layer_c5.bin` | `3886f488d00bb812cdcc1f9d9fd8f6a8c5377ed96542738d1e73b36b4ebb74a8` |
| `insts_q4layer_c6.bin` | `ef533b739abdd58557b0d58bd5ccfb0ec9050cbcaa1935a92ead74c9ae272d3d` |
| `insts_q4layer_c7.bin` | `9b194bf2ff3af60c9598a0ff112dd9e035a4f4175b80b5390162085610145657` |
| `insts_q4layer_c8.bin` | `9c728e892fa4d93a50dbe413cf31015514163fd6d4a778402b45352ca76359ad` |
| `q4head.xclbin` | `eb4967aeb57fb5d713acc7a272770e4ac03e13566be866e84450b6ef93d2613d` |
| `insts_q4head.bin` | `5f49c3a90f429c93cd79d27021d8ce1388e336a0cfd90fb875c0baf6186ec6d6` |

Weight and activation formats are the decision record's
(`docs/XDNA_PHASE0_RESULTS.md` section 4): signed int4 weights (Q4_0's
`q - 8`), one bf16 scale per 32-weight block per row, blocks-then-scales
tile layout, Q8_0-style int8 activations quantized on the core.
