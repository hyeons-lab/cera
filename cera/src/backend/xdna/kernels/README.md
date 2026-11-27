# XDNA kernels

IRON (mlir-aie) designs for the AMD XDNA backend, with their compiled
artifacts committed next to the sources. Kernels cover Llama-3.2-1B,
LFM2/LFM2.5-1.2B, and LFM3-1.7B / d1-2B prefill and decode:

- `q4layer`: a full Llama-3.2-1B transformer decode layer (RMSNorms, seven Q4
  GEMV stages packed into four per-worker weight groups `(q,k,v)`, `o`,
  `(gate,up)`, `down`, interleaved-pair RoPE on `q`/`k` from L1-cached
  cos/sin, 16-token block online-softmax GQA attention, on-core `SiLU(gate)*up`
  fusion, and L1-cached residual adds) in a single run. Ten instruction
  streams (`insts_q4layer_c1,c2,c4,c8,c12,c16,c24,c32,c48.bin` and
  `insts_q4layer.bin` for `c64`, 64 cached tokens per chunk in standalone
  `bo_kv`) share `q4layer.xclbin` so only populated KV chunks are streamed
  without switching hardware contexts.
- `q4head`: the 131072x2048 GEMV for the LM head (vocab padded from
  128256), sharing `q4layer.xclbin` (`Q4LAYER_HEAD=1`).
- `q4prefill_layer`: a fused `B=32` prefill transformer layer (RMSNorm+Q8,
  Q/K/V GEMMs, causal/bidirectional RoPE + bf16 KV packing, tiled
  FlashAttention over standalone `bo_kv`, `O`+residual, FFN RMSNorm+Gate+Up,
  on-core `SiLU(gate)*up+Q8`, Down+residual) in one run. Eleven instruction
  streams (`insts_q4prefill_layer_c0,c1,c2,c4,c8,c12,c16,c24,c32,c48,c63.bin`)
  share `q4prefill_layer.xclbin` via runtime `g_params[2]` KV chunk count.
- `q4prefill_lfm3`: unified `B=64` (`CHUNK_K=1024`) LFM3 prefill kernel combining
  Mode 0 ShortConv (`insts_lfm3_pre_conv.bin`), Mode 1 causal/SWA attention
  (`insts_lfm3_pre_attn_c0..c255.bin`), Modes 2–3 global bidirectional attention
  (`insts_lfm3_pre_phase_a.bin`, `insts_lfm3_pre_bidir_b_c1..c256.bin`), and
  Modes 4–5 >8K paged online-softmax attention
  (`insts_lfm3_pre_fa_part0_c256.bin`, `insts_lfm3_pre_fa_part_c256.bin`,
  `insts_lfm3_pre_fa_fin_c0..c255.bin`) under `q4prefill_lfm3.xclbin`. (Legacy
  `B=32` `q4prefill_lfm3_conv` / `q4prefill_lfm3_attn` sources remain in `iron/`.)
- `q4layer_lfm3`: fused `B=1` LFM3 decode layer (`CHUNK_K=1024`) covering
  ShortConv Mode 0 (`insts_lfm3_dec_conv.bin`), causal/SWA Mode 1
  (`insts_lfm3_dec_attn_c0..c511.bin`), and >8K paged Modes 2–4
  (`insts_lfm3_dec_phase_a.bin`, `insts_lfm3_dec_fa_part_c256.bin`,
  `insts_lfm3_dec_fa_fin_c0..c255.bin`) sharing `q4layer_lfm3.xclbin`
  (single `SLOT_NORM_0` parameter+RoPE packet, merged `Q+K+V+ATTN_GATE`
  broadcasts, packed 4-chunk `FFN_DOWN` activations, and overlaid `DecState`).
- `q4prefill_lfm2` / `q4layer_lfm2`: unified LFM2/LFM2.5 prefill (`B=64`) and
  decode (`B=1`) kernels covering ShortConv, SWA/causal attention, and >8K
  paged online-softmax modes (`q4prefill_lfm2.xclbin` + `insts_lfm2_pre_*.bin`,
  `q4layer_lfm2.xclbin` + `insts_lfm2_dec_*.bin` with merged `SLOT_NORM_0`
  parameter+RoPE packet, merged `Q+K+V` broadcasts, packed 4-chunk `FFN_DOWN`
  inputs, and inline `y_op==11` residual accumulation; legacy `B=32` sources
  remain in `iron/`).
- `q4prefill_lfm2_ext`: unified >32K paged LFM2/LFM2.5 prefill kernel (`B=64`)
  with param-slot ShortConv (`insts_lfm2_pre_ext_conv*.bin`), Phase A QKV/RoPE
  (`insts_lfm2_pre_phase_a.bin`), full-block single-pass quad FA
  (`insts_lfm2_pre_fa_part0_c1024.bin`, `insts_lfm2_pre_fa_part_c1024.bin`), and
  final online-softmax FA+O+FFN buckets (`insts_lfm2_pre_fa_fin_*.bin`) sharing
  `q4prefill_lfm2_ext.xclbin`.
- `gemm_bf16`: standalone BF16 GEMM (`gemm_bf16.xclbin`, `insts_gemm_bf16_k*.bin`).

Sources in `iron/` are cera's own code (Apache-2.0 OR MIT, developed in
the Phase 0b spike under `tools/xdna-spike/`), with `exp2f_16` in
`iron/q4layer/q4layer.cc` adapted from `Xilinx/mlir-aie`
`aie_kernels/aie2p/exp2f_vec.cc` (`Apache-2.0 WITH LLVM-exception`). The
compiled `.xclbin` binaries also instantiate C++ header templates from
`<aie_api/aie.hpp>` (MIT, Copyright (C) 2020-2026 Xilinx, Inc. and
Advanced Micro Devices, Inc.) and mlir-aie / IRON runtime descriptors;
full copyright lines and license texts are in `NOTICE.md` (see
`docs/XDNA_PHASE0_RESULTS.md` section 3a). Regenerate with
`build_kernels.sh` / `build_lfm2_kernels.sh` (WSL2, pinned mlir-aie v1.4.3 + Peano
22.0.0.2026090701); refresh the hashes below in the same commit as any artifact change. Instruction binaries are deterministic: their hashes must match this table after a rebuild. The `.xclbin` files embed build metadata and are not bit-deterministic, so their hashes are provenance for the committed files only; the drift check for them is a rebuild plus a passing `xdna_layer_probe` / `xdna_probe` run.

| Artifact | SHA-256 |
|---|---|
| `gemm_bf16.xclbin` | `c0a004456a93d232b29c8a5644e3b1c5fe28d7186ebc2658376d7249e9fcca2b` |
| `insts_gemm_bf16_k1.bin` | `bbe0a11ce4d5e50c888b759f43e36dfd1b1785f3489687eece82d9ff814ed613` |
| `insts_gemm_bf16_k12.bin` | `b2a7538bae2965ba4fe583d2ee8c086eba35f8b31e3a46451c0eb8e65b264c28` |
| `insts_gemm_bf16_k128.bin` | `0849a3efa4a16de32fbb60d57195561a79a7cf2066ddb242a8389b4cf32bf6f9` |
| `insts_gemm_bf16_k16.bin` | `46e2f679f85d6255c9080a1c72196fd5add2401efed7f72da5bcfa7ed449d8cc` |
| `insts_gemm_bf16_k18.bin` | `f65e28185b7f8af12a5581ee2f93c81897d8581cafbe8f6e775f886b73e668f1` |
| `insts_gemm_bf16_k2.bin` | `b07ccfbef516807740bc834508ae5b865a05de7181a28ced4455809f445a359c` |
| `insts_gemm_bf16_k24.bin` | `65905d79a1b4156c4160f1754b5870bbed51c9b74bb4d61f735389b765b130e4` |
| `insts_gemm_bf16_k32.bin` | `fc5d7db09c8647f9c116838a9fa1fcd126aee0b243fa9d5a0e81c6b7c1501d3f` |
| `insts_gemm_bf16_k4.bin` | `f34556c25b92b412676dcf83df974ad7e98c00fae039e192cf94259ed261147c` |
| `insts_gemm_bf16_k48.bin` | `7f8348df21fa29a02f2b7e3266f5f622dc6bb8f19bf8daf1c73e77c3c436f688` |
| `insts_gemm_bf16_k64.bin` | `4a44185ff9ff04d6abbce19581a5ed6822c6c22667cacc6b4ac1f140718a8fdc` |
| `insts_gemm_bf16_k68.bin` | `c1dfd8839a0c8a7a8ec9d8fd323d0d58ded4be2f4dd440c4ae67a8370eadd8bd` |
| `insts_gemm_bf16_k8.bin` | `b6ca4ab7568c42ebbf37b176d68cfe51fbd6681ee9723dbe407e80035b0a6dd8` |
| `insts_gemm_bf16_k96.bin` | `2953579f49d88b9b42b59de460a9e193afc24bf79ca2de8539552729c31e1d4e` |
| `insts_lfm2_dec_attn_c0.bin` | `619f5eb59df44bca9236630cec471688762f78079c131f0c94bce38e3b7bfce4` |
| `insts_lfm2_dec_attn_c1.bin` | `a6cd2a3ba7c7ea7c7d0d850117b421ece89c81d4f304d0560e8206b9939ac0b4` |
| `insts_lfm2_dec_attn_c12.bin` | `46148fb3ff375c42cc8b416254244d5d4040b128b713b3a640eab8ae7c395243` |
| `insts_lfm2_dec_attn_c128.bin` | `54525fd0b32d8e53ce94ca5a0d8a6b77f680cff9be700c2be43fb88250648101` |
| `insts_lfm2_dec_attn_c130.bin` | `f23160742a29711c80d7eba99ed73f4b12cd8706f01b9efa42551681dce7c190` |
| `insts_lfm2_dec_attn_c16.bin` | `577b0cec8c2315e9088d5dd2db9d564197acdd2fd7e452171c19703fb5dbf6a8` |
| `insts_lfm2_dec_attn_c18.bin` | `286ed14d535cda9572173d159abbe654e007d37ece9ebf36c0f019b12a07fcd9` |
| `insts_lfm2_dec_attn_c192.bin` | `67c8b63a30bc230240640c25e1e3057b0bd4d14d48d6895f1fb09c3b96966b0d` |
| `insts_lfm2_dec_attn_c2.bin` | `6734637d13eaefae92e9e0ec759c6e01d3bdaa069f9726ab8dfbf30d754719f0` |
| `insts_lfm2_dec_attn_c24.bin` | `4d089415043f2862a35df4330ea58a96dd761ac6b611d7f0d19c998edded8af2` |
| `insts_lfm2_dec_attn_c255.bin` | `77581de43f97025f6b170b12129f5261e71cea2e87ee582909c8fe5a6ac91be2` |
| `insts_lfm2_dec_attn_c258.bin` | `11d6ba230c3b39fc8e243720455bf7c5489c4a3abe098d43f965f6206ba9fd6c` |
| `insts_lfm2_dec_attn_c32.bin` | `257fa050078b9c15ec616c929b76098a08f7bdb7acd38df2068883213645fad5` |
| `insts_lfm2_dec_attn_c320.bin` | `3b9b10cfdc46a68fd0dd9ea792207e0e9b635878ea590124eb25f5c6991e8f9c` |
| `insts_lfm2_dec_attn_c34.bin` | `a366584e9b993a59f12b631019b2a16a0e4fa7a3a9e7be5109595bf1126b67b2` |
| `insts_lfm2_dec_attn_c384.bin` | `0a3616347ea960d3e95d25eb4836b921a60ccc212d1a80b8a0642a03c1eafae2` |
| `insts_lfm2_dec_attn_c4.bin` | `ae3223104e92c4733e4464a992fbe724bea003235bdd215c802481d1b5333867` |
| `insts_lfm2_dec_attn_c448.bin` | `a65c7d48885c859f1e6c8194f215fd62a057a841bfaeacd47d9c7706ff402076` |
| `insts_lfm2_dec_attn_c48.bin` | `9059b66651aef6858ff2cb0d9c63600397fc977c8e0810d29854fdd6da8ca7c8` |
| `insts_lfm2_dec_attn_c511.bin` | `ca5618ede0488414b7d8f6d82cec378c3d2026e457fc0026bc7aed0c0125e214` |
| `insts_lfm2_dec_attn_c512.bin` | `25dfed3ab394dcb4362f67f164982a54bf2b90e0637f118fcf60fd5e45dd2612` |
| `insts_lfm2_dec_attn_c64.bin` | `7fbb37736cb1ffedb8f574f8f690c22fc3924805d9da5ff8e04a69e69397c95a` |
| `insts_lfm2_dec_attn_c66.bin` | `6c43c92401f3796c0094e852a879655ca166e0563cb9e81e53ab60e8e238ffa2` |
| `insts_lfm2_dec_attn_c8.bin` | `2f50766e23868cb9ccad1c256291db742df8981792650a1cdad11111d1ad2715` |
| `insts_lfm2_dec_attn_c96.bin` | `ee23cd25ca78a1a2446463ebe311ccf8e5bafc2a914389d1f4c19f1c705b889a` |
| `insts_lfm2_dec_attn_no_ffn_c0.bin` | `fb04f5285dbe86adb456fcfa1ff5e57cdd79afc36d50d0daced0ab3ffc805ef1` |
| `insts_lfm2_dec_attn_no_ffn_c1.bin` | `e93face4af7e82cc165aeccdc9fa0b1c3f3b15ac6b83598d564410c1d882aadf` |
| `insts_lfm2_dec_attn_no_ffn_c12.bin` | `dd18db959e358d9f0fd31eac5315f1c5be343ae446941e87602c406fc238642f` |
| `insts_lfm2_dec_attn_no_ffn_c128.bin` | `1b949ecf4ae603c76e8566be56e4181c23b144cbf9c52a5c6e9f4b67315f15f0` |
| `insts_lfm2_dec_attn_no_ffn_c130.bin` | `f55435f2b6ea8baf94079479ad77016b0bca7197c6773c474a6877e80066cb62` |
| `insts_lfm2_dec_attn_no_ffn_c16.bin` | `8c924f105467d388a8e0ffd5cd34d335f3a0ce9fdf92448a89f8637915081382` |
| `insts_lfm2_dec_attn_no_ffn_c18.bin` | `5680d9808ff488e9dd1cbc532ba032da53788b95b08777a9d4b99bab78dfcb87` |
| `insts_lfm2_dec_attn_no_ffn_c192.bin` | `a1e4a8e246a936eeb996d963c406b7e9a64e893e96247360b007616b2a96a326` |
| `insts_lfm2_dec_attn_no_ffn_c2.bin` | `036b6a34ccbce99bbcc148083b5e3094c96fb56a0312230af37c3303284a6c1d` |
| `insts_lfm2_dec_attn_no_ffn_c24.bin` | `d4a062868a091f08699b5b000ae26358ec5c749bf79c25200e7740c58fbf2fe1` |
| `insts_lfm2_dec_attn_no_ffn_c255.bin` | `3fbd822c5e2ca1d6387046f0e3e018464c68c7aacd2662737ff1c95717a0e566` |
| `insts_lfm2_dec_attn_no_ffn_c258.bin` | `2d1797af7809eb83f9742802a98a2d8522ed7f6a47c414dfbf5d19600d621757` |
| `insts_lfm2_dec_attn_no_ffn_c32.bin` | `485647cadf19146f38e19edcecd996ba61a342e0ce103fa26a573a4afa165312` |
| `insts_lfm2_dec_attn_no_ffn_c320.bin` | `6f07ff8479bbebf0a17b267497e5a57f533ac9f23001ddba5b8f57ec9e4044a2` |
| `insts_lfm2_dec_attn_no_ffn_c34.bin` | `d3390028351b5b748656f656ab779c6d315dc10c1c1b60fc066be0755468c49a` |
| `insts_lfm2_dec_attn_no_ffn_c384.bin` | `dd4031f4dbc2d34cc74dc97726acef923c6ce4f0c5424375519b7a8c9e91ff67` |
| `insts_lfm2_dec_attn_no_ffn_c4.bin` | `73d8cf0f920a201c886860d5604262292026af190d64e9202168310feb175ff2` |
| `insts_lfm2_dec_attn_no_ffn_c448.bin` | `e07440978e76d16b49b8f9d2523423be8ce6ab97f5ece236656b3aedad0af07f` |
| `insts_lfm2_dec_attn_no_ffn_c48.bin` | `e483190b7aaec1f335f996ae3ea15c4b4dcbe86f3c9f414087e5bcdf444e2f99` |
| `insts_lfm2_dec_attn_no_ffn_c511.bin` | `d9a7c282e5110274e9b4f7ded8f15682b7cf7fc61928c784c8ebf6b446cb234c` |
| `insts_lfm2_dec_attn_no_ffn_c512.bin` | `b7d4d94fdbaee68f0480c7286094acf30e9553af99ff2dcfdd722f88cff2ec89` |
| `insts_lfm2_dec_attn_no_ffn_c64.bin` | `98a916b3702656c94a64eb11502fa662db334e593b8b62587cf70cc7c009335f` |
| `insts_lfm2_dec_attn_no_ffn_c66.bin` | `82562ccba922ab9066f550510f9d7eba4983f5516d6b6c70c5674dba9298179f` |
| `insts_lfm2_dec_attn_no_ffn_c8.bin` | `2fc0af767013d47ca966370014a0ce5c70a24be5dbddd50ae615a22bd4b3dae2` |
| `insts_lfm2_dec_attn_no_ffn_c96.bin` | `dd110ccd68937419934cc2a4b4ed9c1dd344d6ac974c9eccfbcc95ee122eab67` |
| `insts_lfm2_dec_conv.bin` | `f1f1db39078f935c52671e5ca68e9c681cd733b771e3773e9170ebd9faa31de2` |
| `insts_lfm2_dec_conv_no_ffn.bin` | `7d6cf7edafdf6e429e49bb275b432b0937aa3f96a27e214cd724a25686c6452b` |
| `insts_lfm2_dec_fa_fin_c0.bin` | `a81ec2f8571f41c923121de2e26588f8d4a2951e1e0e5ba38f82e6a767f86088` |
| `insts_lfm2_dec_fa_fin_c1.bin` | `ee1c961bec7a4133519c4967cff160d4071dcfc02640fceab780d16630d63ac3` |
| `insts_lfm2_dec_fa_fin_c12.bin` | `730121c5dbae02bb88f0c5a10bfd892ec9a8ba38989189df62166b2536268796` |
| `insts_lfm2_dec_fa_fin_c128.bin` | `540429b265636b6b76f6b8ad96f8d59a3446bd3fbfb044c6399ce26d6c48b567` |
| `insts_lfm2_dec_fa_fin_c130.bin` | `8f35eaad1a29ca44e36f5b8df1ab6454f1381168cb77ffbdd0e5813e3defa347` |
| `insts_lfm2_dec_fa_fin_c16.bin` | `8d86122bba6e567a2d6c195820da0383942ee25a451f4351c10b24766a089288` |
| `insts_lfm2_dec_fa_fin_c18.bin` | `f78750938d241f892406a2ad0d5c13f0444314b12d227eb2211fe6f488385fcf` |
| `insts_lfm2_dec_fa_fin_c192.bin` | `c07a9f7149ea1f4e2ccb59bf24716c25d0c1cb45d768e1d3c4119e116ee051f1` |
| `insts_lfm2_dec_fa_fin_c2.bin` | `38beb1d1991231529157ac6ec9b4731fc006b4bf75fa7bbcf2e95254eed3cc2a` |
| `insts_lfm2_dec_fa_fin_c24.bin` | `c7114f9b5756fcf1bfdce3d85c990cb13739f0e7853168a47c1be7218857c880` |
| `insts_lfm2_dec_fa_fin_c255.bin` | `ff670651d7091442d0e830d22349ffac8768e1dee712e975cb7b6f39c1257a20` |
| `insts_lfm2_dec_fa_fin_c258.bin` | `9b490a8cb42b9e677e9b42d8ddfb75f27400a4153e328e232c2abf5438ed93fe` |
| `insts_lfm2_dec_fa_fin_c32.bin` | `851988f3d891d711ca07feea6457f0817e60a29499dd2ecf78fcae5fe544418d` |
| `insts_lfm2_dec_fa_fin_c320.bin` | `58546d0e3bd5905693ed9011069b66ae903f0fdaf1489e2105bac6340b4833e2` |
| `insts_lfm2_dec_fa_fin_c34.bin` | `4cefa9c678bb46b838f69d7e6f5186317ba7f13605f9d187de28de779d1a93bd` |
| `insts_lfm2_dec_fa_fin_c384.bin` | `29c39e5a4b869d474354924eb2c8248e363de0296115191a99ee2ff3821979b1` |
| `insts_lfm2_dec_fa_fin_c4.bin` | `ddcb7e54540adaddf32f3d4ea4102f34f23028360f2669585fd7ed5fbb8fd4fe` |
| `insts_lfm2_dec_fa_fin_c448.bin` | `cd000019aa04c8c29bb0b785e38272a9155b8738c316be3aaf301497389e3741` |
| `insts_lfm2_dec_fa_fin_c48.bin` | `49397a23ed96c811e35746da4ac3c5ceac83ec85cfb87c2f33e6d7596cdcae7b` |
| `insts_lfm2_dec_fa_fin_c511.bin` | `59e4a5f2817fc6a4301794c26bdc79b6d6b898bcaf47c1d49c50a364c14138e0` |
| `insts_lfm2_dec_fa_fin_c512.bin` | `9a0e8acc1edf4f1a8ce6be893468ca33ab6719e26338b3b8b8cd087103fd216a` |
| `insts_lfm2_dec_fa_fin_c64.bin` | `15b641d2a8db7f568c3f29a90e1d26e5aaf53cd8b7096a60a4c24cfb48b47fc8` |
| `insts_lfm2_dec_fa_fin_c66.bin` | `d7ee5151576af148932d2d9861062976938f344c49bbe0904825c1acad98ee76` |
| `insts_lfm2_dec_fa_fin_c8.bin` | `1fd84e11264cbc2f5e70fd2ab8f2a7fec55b6d691c648d1caac9425a8f56d829` |
| `insts_lfm2_dec_fa_fin_c96.bin` | `4e07cd8b02bb7eeb075d0e062748d939f72ccbcd748562039eac95bfa18e0fe6` |
| `insts_lfm2_dec_fa_fin_no_ffn_c0.bin` | `db02b90084134444b9484a6c2e52c9f25b4fb6b79093a9a5465c59bb9d20f35a` |
| `insts_lfm2_dec_fa_fin_no_ffn_c1.bin` | `c277a2fcb4de8b0bb46c57f2f419251b3604258a74b0bf08b48f9293695093c9` |
| `insts_lfm2_dec_fa_fin_no_ffn_c12.bin` | `5302e81a281e3c76af393e6df95077a85c6c6a77eec5b1d0564f4667d67cefac` |
| `insts_lfm2_dec_fa_fin_no_ffn_c128.bin` | `dd2ca8edf90dad683e20365cd996fab747ec58dc234c7e1a98570e1f9864404c` |
| `insts_lfm2_dec_fa_fin_no_ffn_c130.bin` | `da069741decaba9a784ed830e161c2c0b86f8041bf36d3daf2ab79cb2ef78649` |
| `insts_lfm2_dec_fa_fin_no_ffn_c16.bin` | `878bfce5bcc9814b2ff6acea31ba20b8a9b5820b45d327f7f8581e7ebfbac118` |
| `insts_lfm2_dec_fa_fin_no_ffn_c18.bin` | `9f4e96896017330aaff0c2f3d9338c63917c753dc63a22d2206a41d58fad1abc` |
| `insts_lfm2_dec_fa_fin_no_ffn_c192.bin` | `0c61c6f2554ef94601ab90c64d5cfc675e29647184c9ff384540c21c19b72a9a` |
| `insts_lfm2_dec_fa_fin_no_ffn_c2.bin` | `7ecdcd1f69e12a2ad9d667c1ea04deaf486821c8ef056770a45f3ae0e145cbdb` |
| `insts_lfm2_dec_fa_fin_no_ffn_c24.bin` | `775741a1902ff7d67aebc8595f1ea092e02650df26f2ea2934729000ee6e0ac1` |
| `insts_lfm2_dec_fa_fin_no_ffn_c255.bin` | `d0f1f50aca81ba05ada71783638af89085ba3b8571a243d7a5217b71aa09f9c4` |
| `insts_lfm2_dec_fa_fin_no_ffn_c258.bin` | `d21543059eea8373c5f61350a4adf768f27aec996e78e8ba3f91708759bc8223` |
| `insts_lfm2_dec_fa_fin_no_ffn_c32.bin` | `9d97f30cfa83c209d7c11ad58752cd4da7e7d3843269437c0074d33a462c55ab` |
| `insts_lfm2_dec_fa_fin_no_ffn_c320.bin` | `1de87a0992b9cd97dc9b84535ef3cd6b9d78e866046df609770f09d48863b437` |
| `insts_lfm2_dec_fa_fin_no_ffn_c34.bin` | `738bf6f50937c8f83979247baf7beee2c3b4c94b3b78eb336a8643fca4b4825a` |
| `insts_lfm2_dec_fa_fin_no_ffn_c384.bin` | `56d9fd64369cf73bb7d0b67500deb2602ad03b96f4e5ce7117c88ac37d68c3f6` |
| `insts_lfm2_dec_fa_fin_no_ffn_c4.bin` | `c05e77a0506d5ac1316f8287f6a2042089d483152afc60fd251b88ac786fbcc3` |
| `insts_lfm2_dec_fa_fin_no_ffn_c448.bin` | `02305b4a7d72ff034df95c27fd10b103448bf03c6554563bdbd2d913e6298283` |
| `insts_lfm2_dec_fa_fin_no_ffn_c48.bin` | `832c121181e3d926009d989970500fc418304fbe0422e11cd58d83d4618f33f6` |
| `insts_lfm2_dec_fa_fin_no_ffn_c511.bin` | `78b35c21f14dc4448bffcd952a7a5e8c7913040ed2c0cb9fd15b5e897c699236` |
| `insts_lfm2_dec_fa_fin_no_ffn_c512.bin` | `a998c37a6398ae861a6f2b17fbf6f8bec59f1d60d7ad164b05b00f9bcc5e4adc` |
| `insts_lfm2_dec_fa_fin_no_ffn_c64.bin` | `fef62818321d1ae8d919b9c45481c2363e4a4f4a741941ae580f6c641a50b3e0` |
| `insts_lfm2_dec_fa_fin_no_ffn_c66.bin` | `a9a90636771857a6ba79477a2168fcad2a7ba256e60e1922260f4c43e36afd45` |
| `insts_lfm2_dec_fa_fin_no_ffn_c8.bin` | `8c3a294a45867c6c60510d5fc65911d47d03b66b26e06636cfbb9c579433be89` |
| `insts_lfm2_dec_fa_fin_no_ffn_c96.bin` | `12ccaa22bab38e3cbf46e31ad5b2d695928337b62e4f209cd6ede67359241891` |
| `insts_lfm2_dec_fa_part_c512.bin` | `47328bfbaaec5da38f01e85f2c3efbf783e1e8748062958d1b39f8c8226505b1` |
| `insts_lfm2_dec_phase_a.bin` | `5dd326a05d5fad70c4722dbb8d1715f0f37ed8c05ded362057dbfcc8dd4b830e` |
| `insts_lfm2_pre_attn_c0.bin` | `9dc68719a5a29720901527f2847b1131c4a279d2796dcaa5be1a61dda1507e51` |
| `insts_lfm2_pre_attn_c1.bin` | `585925504b7506fca30243d77e45da9e7fdae7f28e24d7e91165765809e4bd5b` |
| `insts_lfm2_pre_attn_c1023.bin` | `feaa832fa7b28b459c80a4eddd7e4bd24de8e00133fc91162798ec1a72a31542` |
| `insts_lfm2_pre_attn_c112.bin` | `ebe1d93a428c084df12f76215273d964f1bb32e09516d70d22baadfefd8d8e22` |
| `insts_lfm2_pre_attn_c12.bin` | `91fd90857b25daabfd7225a9b365e5e476d2f465770b110eb7348b663e0cc00b` |
| `insts_lfm2_pre_attn_c128.bin` | `477f4b836b57fdabacec273cb9fa42ad4a84af3df3442d7465eee80f9aeb54e8` |
| `insts_lfm2_pre_attn_c144.bin` | `ae998a9dd6734ce8cc852abe146adb4034a2da47106446ae3b596a9deef42f7a` |
| `insts_lfm2_pre_attn_c16.bin` | `b3f7a40e912f6c32aaf11d6aa0b0cbdb23340338ac6f32fa123f9e80ce7ef0ff` |
| `insts_lfm2_pre_attn_c160.bin` | `c923da5b25bd5c81ce52a1b3bbcecb727e3e279a2ba0f06fade23fdb5eefd6db` |
| `insts_lfm2_pre_attn_c176.bin` | `93b27011c6f2fcb430b0620ebfb5c4a821add9dcb2b01c73e615c199770b46ed` |
| `insts_lfm2_pre_attn_c192.bin` | `ad67bbffe03efb58faad0a0f40f200077f13c8a437b2ea8f3f08e805e865a9a8` |
| `insts_lfm2_pre_attn_c2.bin` | `1bef18ac1608a94f25a5e41574f2fb8532852588da65f034764413807cc4232f` |
| `insts_lfm2_pre_attn_c208.bin` | `99236d65636e1c5f0239c2afc678b35ff0651ac75878ea57e5dcb409b5b1b258` |
| `insts_lfm2_pre_attn_c224.bin` | `b79f386131ee18ba2bc4a0523c4979089d08aaddd55d5349eee1f4f28950b4f3` |
| `insts_lfm2_pre_attn_c24.bin` | `488a8a197a95222764a5aa71247157c6c3075cf8884a24196c699d16f4fa2eb8` |
| `insts_lfm2_pre_attn_c240.bin` | `758852d3e108c6e63884ade4325b285ead76be4730cec0368d0893dd9ac93f85` |
| `insts_lfm2_pre_attn_c255.bin` | `649eebf38e432cfe69836c764217ea1b1f57aa4027bf8ca285735e26150025b1` |
| `insts_lfm2_pre_attn_c272.bin` | `2853f1cef546a314c1b2792d1f9927a339f7a0125c74bc31c0c8a263119ed347` |
| `insts_lfm2_pre_attn_c288.bin` | `631e575c7df2074fa8e10c2dc2a0bad45e9fccead75706b55cbf562a1f8a8f08` |
| `insts_lfm2_pre_attn_c304.bin` | `8e3c22d5f4fb48a819f7936dae76e6e1d51f8974436ed12e407e01b483494176` |
| `insts_lfm2_pre_attn_c32.bin` | `d372803caae8f88f69f2e6f0eec3a863b9e7c4853898594d86bea3cb2ad7f984` |
| `insts_lfm2_pre_attn_c320.bin` | `e4e68983e77e138d04a622531517b1c0862df09990b61fad50d659fbab9a032e` |
| `insts_lfm2_pre_attn_c336.bin` | `a749b1ed58b5dc05ede1d33944b349267126e0cb7afe8e4ecb82fde95fa89565` |
| `insts_lfm2_pre_attn_c352.bin` | `6b8cab8f6fd845d5ed037dab80bcd46afa3233fbbeb6dc928421ed3d7b8436ec` |
| `insts_lfm2_pre_attn_c368.bin` | `248956d67a8502eb7092d05da6e13680c065c09977c25f2318df1aa8c77c08bc` |
| `insts_lfm2_pre_attn_c384.bin` | `e9dbd49edc1269597629c45f7ad7ae0a19d9eae0f1d8ab9d742bf67e1a40f575` |
| `insts_lfm2_pre_attn_c4.bin` | `7f2f6f4545dc7a316f57c330f570ad6665fd8331a730325e6a469dd53a3549dd` |
| `insts_lfm2_pre_attn_c400.bin` | `6ff7c727242144493139c41d07de789b198c7a4462121635bc47950d2eb15a06` |
| `insts_lfm2_pre_attn_c416.bin` | `fb082bfb7b8518769d47b7a02b4d6697ac1fbfc3cb95a0ce1e650176270e6cc8` |
| `insts_lfm2_pre_attn_c432.bin` | `b914e94ce90c247d949f63edb8156fa46c0495ce135e0f94f84e430abeba7b2d` |
| `insts_lfm2_pre_attn_c448.bin` | `41497ba366345ebd14c6451cbde07e43c6aee670533e64844c69373226f395d4` |
| `insts_lfm2_pre_attn_c464.bin` | `10183a790f6d7272c6da7f446590c9bcd10bc435e08e31cc78bed4cc76bfae27` |
| `insts_lfm2_pre_attn_c48.bin` | `6b1ecd0b359326a8159faac21e9767f272380573245c8ad59506810e7425db25` |
| `insts_lfm2_pre_attn_c480.bin` | `c1a86bc88380b006f79e8aa66cbfc2ac64d2bd0826aaf8413d229dbdab637d68` |
| `insts_lfm2_pre_attn_c496.bin` | `043d58c8aa96b4a0dfcb619b42835866ee96d0b55804b1dcb662d10ef0bbf988` |
| `insts_lfm2_pre_attn_c511.bin` | `bf43e249d669f3b6dd191af5402e20238996dc65bc3b159f002093b13abb9bb2` |
| `insts_lfm2_pre_attn_c512.bin` | `20876c7d9d624a7fa6e9a174e693287335b06d1fba4d97edd9879497b1768610` |
| `insts_lfm2_pre_attn_c544.bin` | `82de31ca906a8352b8d7ec9b79c3056e5039d9b4d6e96a548ff6afadc8bcc265` |
| `insts_lfm2_pre_attn_c576.bin` | `f510d3fb4ab5b803c9151d476f0573e266a02b12158959a8a3ebe0662b132e68` |
| `insts_lfm2_pre_attn_c608.bin` | `f18e3df1507c132f3a177904ca61fda357a03c3ba7953348273b7a543fc35294` |
| `insts_lfm2_pre_attn_c64.bin` | `28210ba31b8686d36e10db2bcfbfc72de8db3dc00e16f6320b7cae85c3ca75cb` |
| `insts_lfm2_pre_attn_c640.bin` | `cbd67363cefb143ad3517cc9f06878d7ab330dfed71be478dad37dc7d8e306e8` |
| `insts_lfm2_pre_attn_c672.bin` | `ee9d399d4b7d9a102de918daedec4fde8239ebd1528e3cfe0765b2e35f0e0fb6` |
| `insts_lfm2_pre_attn_c704.bin` | `197f14fcd091d14f46d21cca9f43ca657276d36e4997f0378e20e9b0b65dbdfd` |
| `insts_lfm2_pre_attn_c736.bin` | `3a8e3ed51e3f5cbe7a5a786522182c1a844d790ae3456601ebe61d9774474e78` |
| `insts_lfm2_pre_attn_c768.bin` | `a396ae5bc1723f6b3317897ad0cc5bd0cb5212e418a9f63de2b9c3fde117affd` |
| `insts_lfm2_pre_attn_c8.bin` | `fd9730c164f709f8ea5a65970447f9f3a1de23d9f015781c25a31278f3bd3ac3` |
| `insts_lfm2_pre_attn_c80.bin` | `aca51fdd42950fc513c8bc40d5ab90f2164714305690c98ad32a9f472d38e4bc` |
| `insts_lfm2_pre_attn_c800.bin` | `dce9626308c6439fb15d3623ef90904c84184c025c27e9dd7671a5017372bdd0` |
| `insts_lfm2_pre_attn_c832.bin` | `bc8f59dfba2ad8d3799200206488be1b3a6f5457f45fb89f087d32019ac95477` |
| `insts_lfm2_pre_attn_c864.bin` | `6c029a567bee23813f7e07d34199ca8be1b54749359dd41866177880c51c3f96` |
| `insts_lfm2_pre_attn_c896.bin` | `152a14d3a215d62364d072369d62d235aafd992cdf1552bb4f23bc4fb5d1ce50` |
| `insts_lfm2_pre_attn_c928.bin` | `1e129250d6b28e0efee25d6c633a93a95cfdef41d90a06bff6b00f4188b808c1` |
| `insts_lfm2_pre_attn_c96.bin` | `3d1259b46de436f63217f6b5797a3dc4354262005feb71130887285dc952a713` |
| `insts_lfm2_pre_attn_c960.bin` | `d278e1b8f7b2d595a56897a85a132b21bfbdbf4c2862959b8ecd0cce7e68d9d5` |
| `insts_lfm2_pre_attn_c992.bin` | `8358ecb584bba7c9c1f2e8823695fa32603f3fd9d8bf6a9267a2023a2e4aab9a` |
| `insts_lfm2_pre_attn_no_ffn_c0.bin` | `7c4dec5a5a8b59366681e9b0daa3b5305beafcedc3dc128b977b3a616c676661` |
| `insts_lfm2_pre_attn_no_ffn_c1.bin` | `76cc8dddb1ff6ac9f0ca54926fafe20e1d02762ff8cc1b6ee7a4dabd3dad2f13` |
| `insts_lfm2_pre_attn_no_ffn_c1023.bin` | `dd4f9d9aef121176a2c2deb26f6461a9ce53359941b1477417f2282e39a0a345` |
| `insts_lfm2_pre_attn_no_ffn_c112.bin` | `5b3c7387f0c43b9ce084eaead38016d6f38d9a42e00ed9f4cfb60a4b78f3db6f` |
| `insts_lfm2_pre_attn_no_ffn_c12.bin` | `697e8b90a91312b1e9980b6303d27ed0a827ca48e30804decd358af2d44e9437` |
| `insts_lfm2_pre_attn_no_ffn_c128.bin` | `3e4e25cb1f1d594e65aac5963bec40f9ad2f4ce22eb50d0665d669cca8a1d2c1` |
| `insts_lfm2_pre_attn_no_ffn_c144.bin` | `dc9a67a3f1741dd22ba050f2f7ac7f801eca7f71131ae91dd756278de3fc613f` |
| `insts_lfm2_pre_attn_no_ffn_c16.bin` | `fa9f3cdc47a9551f05d8a9b15fd27c856b8dc549c3174e7f9c370d4ef097f074` |
| `insts_lfm2_pre_attn_no_ffn_c160.bin` | `3ed182d23a5156c57830bb375eb65c203620fc9f94c3e7e642c135f7d9496a67` |
| `insts_lfm2_pre_attn_no_ffn_c176.bin` | `687258c55e0a1a587d4836f4675144460f2c2115df294a98522d6b87f3d4eff7` |
| `insts_lfm2_pre_attn_no_ffn_c192.bin` | `e08bfeb59b82d4591c1624dc42a61767ecf107d0237b4c158b8c84c7052ff5ef` |
| `insts_lfm2_pre_attn_no_ffn_c2.bin` | `bbc055ae6d11783f7cf539fd5ebf3efae0d36ca94380287f1325c179ab2e9779` |
| `insts_lfm2_pre_attn_no_ffn_c208.bin` | `0cbb576627dfdd1034ec9c9e919a79f59360b57ab888cbbaf0cc0ff4a1f9753c` |
| `insts_lfm2_pre_attn_no_ffn_c224.bin` | `3058cb5cd76d1f3f289099c406e4c1ec5eee23d9bdb4b1fd2f26ab71d7972be6` |
| `insts_lfm2_pre_attn_no_ffn_c24.bin` | `4ce2fda91d547e957f19560df7ad38bd40bbe0f857c38ab21b724c1545770f48` |
| `insts_lfm2_pre_attn_no_ffn_c240.bin` | `d4b6a886d7bbebf947071cda88a03946b4b21f6b887b2855c9e8a5d768a006aa` |
| `insts_lfm2_pre_attn_no_ffn_c255.bin` | `6c0c7b211b9b3f6e01337a457937045f72c2c0d2b007ab1a7fb05d657905c8a9` |
| `insts_lfm2_pre_attn_no_ffn_c272.bin` | `ce63c199b7b832dca3fe42fc4e450d8f71508ee09f1dd43fe3b0fda6085f3e10` |
| `insts_lfm2_pre_attn_no_ffn_c288.bin` | `635c8c710cef51cc91bae21eb88859bc78d48327856d8db7765674a8cdbfdea0` |
| `insts_lfm2_pre_attn_no_ffn_c304.bin` | `d4166fdbf8f271e63eb01666a491e51aa38d2e0fea2f6f6b8c1a4561fa5be56a` |
| `insts_lfm2_pre_attn_no_ffn_c32.bin` | `3f2ff027e05d0a9f9fb93112c33a47823fa7f8288161171d31b95cee70770894` |
| `insts_lfm2_pre_attn_no_ffn_c320.bin` | `bc90c79b2512b863a9cdc0bf3193cc31497ed38a13dd7c59b60e94deec24bd2e` |
| `insts_lfm2_pre_attn_no_ffn_c336.bin` | `37d782ba8366e9c7b0a5c757041f6e94d6d3d18f029e85ad4af7618e08337920` |
| `insts_lfm2_pre_attn_no_ffn_c352.bin` | `d8e3f664d74a053e72fcf18e4ca01ef29e93b8c1583e58ab6fec25d74ea58903` |
| `insts_lfm2_pre_attn_no_ffn_c368.bin` | `53382440e7712962e9c1ce57e5f60d48fb952ae1a72fc35dba478f8071a2ddfb` |
| `insts_lfm2_pre_attn_no_ffn_c384.bin` | `ae8dbfc890cfe78dc421c50186729997a15db985e916f4f8b0508582ad5b5821` |
| `insts_lfm2_pre_attn_no_ffn_c4.bin` | `d629e628c70b432f9c5847ab2dde17aecd1a022962b41b0c3e7b2f0972bb5321` |
| `insts_lfm2_pre_attn_no_ffn_c400.bin` | `8cbc3a23fcedd59aca05d8efb582f22580245d530535666704ec1ac103f6cfc6` |
| `insts_lfm2_pre_attn_no_ffn_c416.bin` | `ed7e50b1d46bd5654ff9249c5f5f5de4ef9fcd126119f85179c63a0aadd702c0` |
| `insts_lfm2_pre_attn_no_ffn_c432.bin` | `4ac8df35aff801dae2dfc06596610443c9c5df46bae70d5b5880df428d53662a` |
| `insts_lfm2_pre_attn_no_ffn_c448.bin` | `6894555ca107611a919d0d21413fc5173e439ffb2861afc13f0a14e8a409e1c6` |
| `insts_lfm2_pre_attn_no_ffn_c464.bin` | `671b4b808dc31f75730db7d615fc6f5db35806574d81f10b10437326ba51be8a` |
| `insts_lfm2_pre_attn_no_ffn_c48.bin` | `650e4b9b7efe21b85d105b972808982fa5fb1796f12f14214dce18304495423b` |
| `insts_lfm2_pre_attn_no_ffn_c480.bin` | `29039aa0ec724613dc4c2f8bd3fdd2103b2d4dcadbcb5fc3728c9126ab908bee` |
| `insts_lfm2_pre_attn_no_ffn_c496.bin` | `2d70039837b08aa87d5eb318a4b143ceb10e933e970d88008f41f1c9274bff9a` |
| `insts_lfm2_pre_attn_no_ffn_c511.bin` | `3cf350f19979f29cc84acc34e385ffe81999c17c90e164bcce22c72f8205121b` |
| `insts_lfm2_pre_attn_no_ffn_c512.bin` | `0b170ee037d5f28a3a16b41efaf3957230864763fa1718f58f0e5c3560c4d9e2` |
| `insts_lfm2_pre_attn_no_ffn_c544.bin` | `3ddfab282a1422d0d7047bfef0f6acb187f41ebec4f999a67966d1999e4fa072` |
| `insts_lfm2_pre_attn_no_ffn_c576.bin` | `3fc7621dad29a9da91e31bc5cf83f8e6ce7c702abb5e2025c224c359045db564` |
| `insts_lfm2_pre_attn_no_ffn_c608.bin` | `511aedd102746f4e045faf95f225358d810c3337f25ff66e06032148ced16ed3` |
| `insts_lfm2_pre_attn_no_ffn_c64.bin` | `eeb20ed1b121af0b10876c9f525954eb33336b96bb0063859a76784c3ddf9845` |
| `insts_lfm2_pre_attn_no_ffn_c640.bin` | `232aad47b4270c65e61ee19c99e7790116c72d211694974e0abedd94dfa12695` |
| `insts_lfm2_pre_attn_no_ffn_c672.bin` | `41e8b6a26cd0081315c6cabe121733bca64c98439877913c00b0a8c123bead0d` |
| `insts_lfm2_pre_attn_no_ffn_c704.bin` | `c5c088b63eadf585c1c35e0c7b0d0a8d56ac1e92a57ba4f2aa3b50eef496567d` |
| `insts_lfm2_pre_attn_no_ffn_c736.bin` | `d513cd6f07ecded727b049658e81d737a8282a2df6a8702cfe7868eb4d639d2c` |
| `insts_lfm2_pre_attn_no_ffn_c768.bin` | `92b03f4fe8b0ea6a41f49d98bd499835b343cc13efbf5c44c01941fe43dc56a0` |
| `insts_lfm2_pre_attn_no_ffn_c8.bin` | `8552ea308169b46c22ada4f7ee52a198f9433a7042c2b91ec95488dfeaefc621` |
| `insts_lfm2_pre_attn_no_ffn_c80.bin` | `efcdf82d85563ab2fa04affa07847c3480d0bac1f55dd0479e549687f1427eaf` |
| `insts_lfm2_pre_attn_no_ffn_c800.bin` | `74395500d07d3f6e39e7d1b2fcdf4e898992e106e1352cb071372dc5a8af264f` |
| `insts_lfm2_pre_attn_no_ffn_c832.bin` | `887dbdcd22325530586d7e2d133b68b6aa7f1fd8b4cb5d78165256d4b7c84ea6` |
| `insts_lfm2_pre_attn_no_ffn_c864.bin` | `0591d13763a7fdd84fdb66f96994f8910c734b1398242dc7855a708ccbe4fb04` |
| `insts_lfm2_pre_attn_no_ffn_c896.bin` | `bdb28a5c1a6f0fdf5ad4ce4a08be1f2dc892c0eb0491d152b67c1580a5785aa6` |
| `insts_lfm2_pre_attn_no_ffn_c928.bin` | `761f397956ea47c395507c11cd30b273a853632a66fba465deff3b27ddd4d930` |
| `insts_lfm2_pre_attn_no_ffn_c96.bin` | `fc080e37149b181ac134b59519f79fef8e6fab7ab032501a39be6db766a4225c` |
| `insts_lfm2_pre_attn_no_ffn_c960.bin` | `a7e5faed0e577d788cfde954ce4730890208ce15d24ab176eaa752b5a184bea6` |
| `insts_lfm2_pre_attn_no_ffn_c992.bin` | `9faa63c2b8a9441d97af24f0fee8917cba5eb9328ded76a42a3df85e939ed50b` |
| `insts_lfm2_pre_conv.bin` | `35a02f0cd597c17f978948204b0f9904f719bc7559036a1b62a9b48747155919` |
| `insts_lfm2_pre_conv_no_ffn.bin` | `152577304d7f85b90dd7a8a57344e9f2165143947d4ca732f1081be54a22d50a` |
| `insts_lfm2_pre_ext_conv.bin` | `68f89580ecfb72347abc4a774050fdbc74c7fd16028f760c1598f96554a934fb` |
| `insts_lfm2_pre_ext_conv_no_ffn.bin` | `69ecf560e31a1c75d48f063601bde54919096236096165c64850a96449a18ad6` |
| `insts_lfm2_pre_fa_fin_c0.bin` | `6ed5847bb2cd77683631f388dd8598dd3a9131964b184684c3bc948586575c4d` |
| `insts_lfm2_pre_fa_fin_c1.bin` | `364d20a2d5c36179abc11691d2d292a5d623ab59f9125af9c70c74d1d725ebf5` |
| `insts_lfm2_pre_fa_fin_c1023.bin` | `49e43a3d344592cecbb38dc81a27c3729f99a36de144eb377448e6e672eb83bb` |
| `insts_lfm2_pre_fa_fin_c12.bin` | `ba56d2e46e170f4258dd62a3de3b4776382a1d2a12528c37b26d84def123ad9a` |
| `insts_lfm2_pre_fa_fin_c128.bin` | `69756bea65603994a4d64cb3be5de5025304d3ad83919198b3d7891e589ef980` |
| `insts_lfm2_pre_fa_fin_c16.bin` | `5b75ed0ecd63c7f4c1b0237cb48d8c46ec1e109c8c8c2c8455222086f3ccd28c` |
| `insts_lfm2_pre_fa_fin_c192.bin` | `c064ddd8aa14772d2e6c521f5391686b57535896c470098c17f78b6b832f2729` |
| `insts_lfm2_pre_fa_fin_c2.bin` | `5b3ba4e9bd2eb052e6884c03578224f5e5ac72ae60c3178708bbfe3f531405fc` |
| `insts_lfm2_pre_fa_fin_c24.bin` | `ca0c6e84f97f71d61a2e86fbabee48a06460619da05dc23ff30fb8bdf3f617b9` |
| `insts_lfm2_pre_fa_fin_c255.bin` | `0dfaf59a13c5103120626b24ca9049acff4580b8283de494a58326e705da7b49` |
| `insts_lfm2_pre_fa_fin_c32.bin` | `9678bfa8b917dc9bacc697bbb46005d5bcdb2efd1dd0de8a76e7c5fe297c30ff` |
| `insts_lfm2_pre_fa_fin_c320.bin` | `09e2501f6678210420845c7e519cc840d5c9034aedd413569b21cb86aa098a30` |
| `insts_lfm2_pre_fa_fin_c384.bin` | `fa3c1222312dd0250d0e717626e49d2d7703a008d5612a1fbc34723a8c084f03` |
| `insts_lfm2_pre_fa_fin_c4.bin` | `649d11a99fb38d126d35b18efb0f937e27cc948e0d7bcbe753b7aca381c50c3f` |
| `insts_lfm2_pre_fa_fin_c448.bin` | `ca28dca01503a84ccd21b5a0c16391f6cebc25b2415c5262b1be436a8568ea89` |
| `insts_lfm2_pre_fa_fin_c48.bin` | `c78cf85f575094b891f21365e4b0feae36a30ef7eba1bab8c6e9e5db53e1e623` |
| `insts_lfm2_pre_fa_fin_c511.bin` | `6245f49e5e3814baf361f79452028a54c7ff309712dac93f97fd4597e16748e9` |
| `insts_lfm2_pre_fa_fin_c512.bin` | `3def7802d8828903bd169bc4d9f1047d69054250aace9894bf3fa79166586eb2` |
| `insts_lfm2_pre_fa_fin_c576.bin` | `479433b57380cd0172030668a1edce99553118878dd828f645be644e454f750c` |
| `insts_lfm2_pre_fa_fin_c64.bin` | `400ad588336796c0fa7fded88905e13ce5937ce9b4a03e5df9d38e1466ea2c9c` |
| `insts_lfm2_pre_fa_fin_c640.bin` | `1fe7420f28c017cd41f99c23885516807496ecfe526c3b5b2f2219de6136a9b1` |
| `insts_lfm2_pre_fa_fin_c768.bin` | `149d0407b92e74c5b21941dc263718912c4c3e4a7db9b85cc6255020bb449ca9` |
| `insts_lfm2_pre_fa_fin_c8.bin` | `45f8036aba2edb2f5239abf453cab65f0e72f65a89ef5fc82cad6900f5a520c8` |
| `insts_lfm2_pre_fa_fin_c896.bin` | `1c779ba5ec66fbbbcb7ea700b84c09d58bd4e9d5ee1056d3909e69b4ca18dfd7` |
| `insts_lfm2_pre_fa_fin_c96.bin` | `ca051b874cef2d98619cfb7167e8afb06a2be5036343e0b087f60cdd96732658` |
| `insts_lfm2_pre_fa_fin_no_ffn_c0.bin` | `821d76a6008a4472ac296e7c9ee71907424e4e5dbb90864513b5cfd3f5bd9730` |
| `insts_lfm2_pre_fa_fin_no_ffn_c1.bin` | `078d28abff42c54cd6c37df9d6db5676e9ff27be25bc95cd00e5c1e84b003935` |
| `insts_lfm2_pre_fa_fin_no_ffn_c1023.bin` | `79a4cca8a58bc4d56a56c94452cdfc2ad9b9835fdb6757ae715ab1a4a5a45073` |
| `insts_lfm2_pre_fa_fin_no_ffn_c12.bin` | `37ca782321ee6647011a2714a8f8085669bea239ed6b2984c5b574f4725b0159` |
| `insts_lfm2_pre_fa_fin_no_ffn_c128.bin` | `43c20f4d389f6abdd460a7e7000b87dddd7faf78a3ed28df3a50e6a99268c21d` |
| `insts_lfm2_pre_fa_fin_no_ffn_c16.bin` | `4f18228250b9d0bdbbe1548ecd81dd1f13291381f99f9a0deba5f8210af2e186` |
| `insts_lfm2_pre_fa_fin_no_ffn_c192.bin` | `ff78054411d9dde1f56f41e7fa6911c19ac021bca6483004f23bc99741a0d1ef` |
| `insts_lfm2_pre_fa_fin_no_ffn_c2.bin` | `74a81c3d284d53834504ac43db49c45b1a7c144d29e33a3c58bd953bfdd72002` |
| `insts_lfm2_pre_fa_fin_no_ffn_c24.bin` | `7107263fb35e105f96b2a70b654d9779a4870c9734bbd1fc5e30a85296d46c28` |
| `insts_lfm2_pre_fa_fin_no_ffn_c255.bin` | `27ba635a19e28edc214198a885af7c706f0ed469475cd3db012f8d62fb28142e` |
| `insts_lfm2_pre_fa_fin_no_ffn_c32.bin` | `77953a2a7c6e87abe38403b8ace2c13370ca17af8197a3e4597214cdd0df4481` |
| `insts_lfm2_pre_fa_fin_no_ffn_c320.bin` | `5d10a169e97787a2c6f1927b6533457b4ed1a47b84c4d3a7a094fb8a1bfc9f50` |
| `insts_lfm2_pre_fa_fin_no_ffn_c384.bin` | `96b3c63bc6bee02bfc864af214b8021f4c262eb3b6f73c853d157ea15882ad65` |
| `insts_lfm2_pre_fa_fin_no_ffn_c4.bin` | `ff2833914519ea127dae4c624a86c637543885829e32574023089b8d1518f89e` |
| `insts_lfm2_pre_fa_fin_no_ffn_c448.bin` | `61b2719a5f2b5081dfc334a6fa9c6ab01a5fdac7766a90a4bf5b6f6309b07263` |
| `insts_lfm2_pre_fa_fin_no_ffn_c48.bin` | `79e8109255e008036a991c7a8f84c2d2b4ee5a93df1bc546d19ab434c80df7fe` |
| `insts_lfm2_pre_fa_fin_no_ffn_c511.bin` | `517e4b983077751dec786a18d7a02f2d1f76a0513e725c898eea914e2cbdc43e` |
| `insts_lfm2_pre_fa_fin_no_ffn_c512.bin` | `61e787bc727ad5d034ad642539b6d7f0be9ec970270227a1d98ad7aaecaf0f16` |
| `insts_lfm2_pre_fa_fin_no_ffn_c576.bin` | `08bc8208524e309db7b68cfc9d0b13eb336b00f5b9b3eb266fe2b29108c3938e` |
| `insts_lfm2_pre_fa_fin_no_ffn_c64.bin` | `194872770e4c4b1802aa4c1063f542bf5a319489d7596f421aa9b8b29663c3b7` |
| `insts_lfm2_pre_fa_fin_no_ffn_c640.bin` | `753ebc8b0748ae8db24a749cb155e23863ec77fdd267619a56bcb69210e2c137` |
| `insts_lfm2_pre_fa_fin_no_ffn_c768.bin` | `bef1440776bad796d01034964cf986aac56e744650028f0b4f2ada56a9afc5da` |
| `insts_lfm2_pre_fa_fin_no_ffn_c8.bin` | `8b37e3828053208908746f6b07c27b5f6c22fa2521256f270c2b6e41d1e4f533` |
| `insts_lfm2_pre_fa_fin_no_ffn_c896.bin` | `d0ccf5f072e121ddc2a32ffab7f0dcae73e696e32065fce836599ff990c81488` |
| `insts_lfm2_pre_fa_fin_no_ffn_c96.bin` | `95ca7dbedc6eaf38576e29ecb91126f3141663554cf4c750671df90c6f7604f7` |
| `insts_lfm2_pre_fa_part0_c1024.bin` | `aff32588fcf708125e340d87ad33cec0b2dffcf6f802c68954395eaacae843a8` |
| `insts_lfm2_pre_fa_part_c1024.bin` | `de1536de4b7d3f06ec3d848c54a2bff90ceaf51dc0286d055edf518f91614c2d` |
| `insts_lfm2_pre_phase_a.bin` | `3c9d4a3c1f6f0da2d05ed45605379ee79910264a36cd44a8bf3cd1d98a2969fb` |
| `insts_lfm3_dec_attn_c0.bin` | `dcb48858fa6694a800cc800857f7b4b54f3104619bc992f55fbd55a32f880510` |
| `insts_lfm3_dec_attn_c1.bin` | `0b1fabd681ce5ee386828cd69dacade039d1cebdb004a88ed2448ab54fd410ea` |
| `insts_lfm3_dec_attn_c12.bin` | `079c3e66594faf84aa2ab796b754abfb379b307d3c72773513d333aa9b1943f8` |
| `insts_lfm3_dec_attn_c128.bin` | `0a07d357d1de40b5ad229842841d59c767b4c0aade6e25e7cdabe7545eb984c6` |
| `insts_lfm3_dec_attn_c136.bin` | `8080cbc607e7f30bc1a7d9a3fe0aba39e0932e121474ac3658ca14cc218f94c8` |
| `insts_lfm3_dec_attn_c16.bin` | `86294c5ea29f6ea5b4304ca04626753d1a040d9775639440006bd661d32f4976` |
| `insts_lfm3_dec_attn_c160.bin` | `39eea424674d62e1ed85f5f7e49c362e34547565d296b11538870ad8d661acab` |
| `insts_lfm3_dec_attn_c192.bin` | `5871ba624d8731342e521244a4f62c6ee2183c723948051321d63a5a8197cac5` |
| `insts_lfm3_dec_attn_c2.bin` | `9567bbca0c84b1c7145a3a6b61f1365679abedbdb5c03b480fc5dc7e2c766458` |
| `insts_lfm3_dec_attn_c24.bin` | `607df4c6e188384f337f8fd6a8ff96c1f9d4989c9af843c9fa4221e8cd767b87` |
| `insts_lfm3_dec_attn_c255.bin` | `46d5e9bb03203e8281872c51437b1e0b3c0b303dee675ccaccb310fbf1933b31` |
| `insts_lfm3_dec_attn_c256.bin` | `8e1eee89ef95cc8b17d75a171fd1fe2b8a9403c8dd4f63969d34b60aaeedd448` |
| `insts_lfm3_dec_attn_c264.bin` | `b5b7791951d19404c4d635f1bd0ce625a228aa9d9e368a7bba92ccb03f1cd798` |
| `insts_lfm3_dec_attn_c288.bin` | `ec0ad9fd2c9699883d83047f25c6404829c2dd4a7c5ee8136e6ba02571c8f295` |
| `insts_lfm3_dec_attn_c32.bin` | `7ae7f49154045aa46f5f009fa52210f8f1fe47824b6f6a1dcec42e2de97d2440` |
| `insts_lfm3_dec_attn_c320.bin` | `1542cf759e8aead639d397225c09cbabe509d0ee52039b7aba11a3ea1e3a1fd0` |
| `insts_lfm3_dec_attn_c384.bin` | `ecd631e9f0304778c201afcc61e16ad354e2a6ba2faae85fcac9292dd54044ab` |
| `insts_lfm3_dec_attn_c4.bin` | `1220b58567ac64897f0d5735b99211d187db2a33cc25325a40260ebdb65797de` |
| `insts_lfm3_dec_attn_c448.bin` | `fd8375de02b5690adf184b832d9f885194a1541598bd92f90a85be7a5e776602` |
| `insts_lfm3_dec_attn_c48.bin` | `aa8548f6360f457b4ee1b36b2c4a5e5712e0295e3daea3ca845a4a8f4575a213` |
| `insts_lfm3_dec_attn_c511.bin` | `fa9d6a1a295a39da192b0788fca34dcb637876d8cae51b36e10cf22968ad6758` |
| `insts_lfm3_dec_attn_c64.bin` | `7c9384f945d67e13e69ddd37750096ea244a5d3c3aba385191f5e5a633b19ca7` |
| `insts_lfm3_dec_attn_c8.bin` | `c15e32ceb203dd7d9ca7b7b28c315f18692c9b90b9aa8c82d72bd094ef82977a` |
| `insts_lfm3_dec_attn_c96.bin` | `97f83970e1f7bf4642658ed51d40f13256988a34055ee19c34a8db4392c4d8a1` |
| `insts_lfm3_dec_conv.bin` | `48eaf6d3ba87b874c9062d890481f8a7a10b089d4bb642028d149b7a6f927537` |
| `insts_lfm3_dec_fa_fin_c0.bin` | `8067bacadb10a7000add6c45e2599ea27a819ffb4fe1b29357f25acbc538f7f5` |
| `insts_lfm3_dec_fa_fin_c1.bin` | `1e1e60eadfc78a46c241473664e8041b4ed8cb45239430be753f8856d63fa262` |
| `insts_lfm3_dec_fa_fin_c12.bin` | `a0640b4c1cf8b3e97ee4054043870847e4ca5af90f9b36ba3559f7c4e43812bd` |
| `insts_lfm3_dec_fa_fin_c128.bin` | `62d8c55ba64cfcd841635add8dfabd0cdea635681d2e76f537043d87b7f7ece0` |
| `insts_lfm3_dec_fa_fin_c16.bin` | `9e6d13d7048a978057d1b5f5f1847f3d203f1ff97428eec7bcaf9c28b4049701` |
| `insts_lfm3_dec_fa_fin_c192.bin` | `e185f69cc1df3bf48f7a3a95e32884896f8448b3bd6a9f3f58170ba2122c16e1` |
| `insts_lfm3_dec_fa_fin_c2.bin` | `a91158625db7d26df92f5f16bbdf3ecd60e1ea65a887dc7932eb7b783dce1299` |
| `insts_lfm3_dec_fa_fin_c24.bin` | `d583135aa67930d84cf6f15b1633a23977414bbe9415078ccdb2c20032b1390c` |
| `insts_lfm3_dec_fa_fin_c255.bin` | `f3b8a25770f2fc796e42620f8f988e2547f4f1e9ee8a88abf1fbd76e2f65b2db` |
| `insts_lfm3_dec_fa_fin_c32.bin` | `5e6fe2914d153b3d4fedc84951ea4255c6fa7f982bc7833823028a5bb340aec1` |
| `insts_lfm3_dec_fa_fin_c4.bin` | `abd6aa7bbd0cc74b54740e4e4cf7674e0d37ccc139dac48e1550f316f664cedd` |
| `insts_lfm3_dec_fa_fin_c48.bin` | `cae9ea2168739c8ae14e21a78bb4a4f32ab75aaa70afe393d0947b090f3663d1` |
| `insts_lfm3_dec_fa_fin_c64.bin` | `e8d79bdd4ab2a766da21da5ec506ff244088eaf4b6d2e874b3a4a9334e328cb2` |
| `insts_lfm3_dec_fa_fin_c8.bin` | `25abee92989f4ce2a9c75195333a44898117e47011f1472b06bf127db42cbdaf` |
| `insts_lfm3_dec_fa_fin_c96.bin` | `f6623f5746a10100cfb287ec28c9db86fe67f608e19545a76def0de52eb5bc03` |
| `insts_lfm3_dec_fa_part_c256.bin` | `54cb6c416be9bd72e82a8d85be00d3624bea69ce2c39ba879e7dd70b605e079e` |
| `insts_lfm3_dec_phase_a.bin` | `877995d95618f97c6818397bac41a184bcc7b547adc4f5783dd87bd30b5d0353` |
| `insts_lfm3_pre_attn_c0.bin` | `1d526009b01dc91beba2cd67368da99eb7c2e8e3afd9ccdf39de5c8a5e3b1a6b` |
| `insts_lfm3_pre_attn_c1.bin` | `2a188a28bf294f77953218a73c8a55a34612a08ec60116346dfac3c5bf553c01` |
| `insts_lfm3_pre_attn_c12.bin` | `ea8f08d62a0c4c097c853d99205401bc0ed7bbd512b78ea45defe1e7d6fab095` |
| `insts_lfm3_pre_attn_c128.bin` | `8d9a4b8b2b4cbbbd54af43152e16398c42bf76483f355ad5cd508ccd94497743` |
| `insts_lfm3_pre_attn_c16.bin` | `21a0fbfa91e131fd2d65246c8fbbbe95c74c567e572178107efa77c482df8c94` |
| `insts_lfm3_pre_attn_c192.bin` | `7ef4eaadcca1975991b5c0b90a215f7ba5c3a6e696805d6ec875bdc0a42ea06b` |
| `insts_lfm3_pre_attn_c2.bin` | `1853d79c03b1fe007c3e291a5fe0b9222230886a07e80fa02e3da39d28ac855b` |
| `insts_lfm3_pre_attn_c24.bin` | `603ca1dd4d28d885f3bdb5dc6b9c9dac490fde5950b7e8f2c7c713ad47aaf447` |
| `insts_lfm3_pre_attn_c255.bin` | `c5076232cdb642c3183edd879b4e28b9495eb443b1dd0b4bb49ed97091675ab3` |
| `insts_lfm3_pre_attn_c32.bin` | `3d0b9d2f6c4290c473573d38a8d1f911fb46e900c47f57de7ad94186c1d55e9a` |
| `insts_lfm3_pre_attn_c4.bin` | `d68e62e8a003426205b9c65443986af2f02841a4df1ac0aca554c55c32f2c2f7` |
| `insts_lfm3_pre_attn_c48.bin` | `b7a3a0fdfa9b87d78094ceb73b52b2f3003712b834fcdfeb83575299832c06ed` |
| `insts_lfm3_pre_attn_c64.bin` | `7bc1a3a13283994c174c8347722b7112c1ed856bcbe1dfc71d800a849f6183c7` |
| `insts_lfm3_pre_attn_c8.bin` | `7be87e4ce9b7e467aee4d1d5b4cc627f0c346e8f9e395e17ee4b98150181b127` |
| `insts_lfm3_pre_attn_c96.bin` | `44e7cfeadf81ec95d9a09bc8167ffcb89b47afbed19e5226e48ce301e8c15455` |
| `insts_lfm3_pre_bidir_b_c1.bin` | `809831178acd3926d619b280b56c47c10587eafc74b86fed1da1ec01fe94c2e6` |
| `insts_lfm3_pre_bidir_b_c12.bin` | `1a970d53130150197d6254392c9380c480cd2fcbc52f0a8aa7097ee34ba696bf` |
| `insts_lfm3_pre_bidir_b_c128.bin` | `bf523e006d254e9837673ddeb5e995fd08349790c94715696d66c77160867937` |
| `insts_lfm3_pre_bidir_b_c16.bin` | `7bad5121d6a8648ee3b0397a24a64fe9261c50772b35fd5afdbfa216b19db845` |
| `insts_lfm3_pre_bidir_b_c192.bin` | `c44300fc5cff408805eecf45dc2e574695e48e9b6c75d6fc02340d678c635a2f` |
| `insts_lfm3_pre_bidir_b_c2.bin` | `c6a6e7af20f4068de583cc83a007e5b12cfce188676f35211e73c5ece0dc527d` |
| `insts_lfm3_pre_bidir_b_c24.bin` | `5220d6350503a90735412c25ae1992a69a414d20efd3bea580d6d46a106d113a` |
| `insts_lfm3_pre_bidir_b_c255.bin` | `3428261d42c252228cfb7d6b49a1729da9ddb310446514685753d2f640be2c49` |
| `insts_lfm3_pre_bidir_b_c256.bin` | `8735c024b3b42f7785d06e9cfbc8d1ea0c18f044086049411c99593555ecf226` |
| `insts_lfm3_pre_bidir_b_c32.bin` | `c29e960422c744433a0b9e1f1b7a6b867085f8be37179997bf3da937084303e8` |
| `insts_lfm3_pre_bidir_b_c4.bin` | `6e9ab6a5b8c9924c4a396ea2c3603ea6c69b9c2ce782e5c75003239a5046edd6` |
| `insts_lfm3_pre_bidir_b_c48.bin` | `53f386cd92119592bc4b013e0b70b6f1f91e32c6b218db23940f2c78ea2d835f` |
| `insts_lfm3_pre_bidir_b_c64.bin` | `d5f4649191d28ce520ea138cfe4aa10bf50a446fb73f9bb0c54137919063fa1d` |
| `insts_lfm3_pre_bidir_b_c8.bin` | `f75042f12f21eff47e7f060120e6d9cc2e69a7210a7252fe3241a1e3cce45751` |
| `insts_lfm3_pre_bidir_b_c96.bin` | `9bd1fed9a3aa49413f97624183e4774cef0842186700b7440de4d7015a582056` |
| `insts_lfm3_pre_conv.bin` | `edd961f66dcca35b522c6ba2da8300ed669b663a091f5039958ea25727ca9919` |
| `insts_lfm3_pre_fa_fin_c0.bin` | `212077a2eabca35be352822533c187f1f12bc822de12c7d3f0a1cf93cb3ccffd` |
| `insts_lfm3_pre_fa_fin_c1.bin` | `87ec27387ff4bfac3ced842662a067065e354cfc7d227b0fa42171079abacdf6` |
| `insts_lfm3_pre_fa_fin_c12.bin` | `d6a74addbf5d8e5b4e0723f851a9dcd2769a9dcc08344fad85b34f74f5afd48c` |
| `insts_lfm3_pre_fa_fin_c128.bin` | `099364746c4a2e536aefbaf86445c7449c97df4078fefd51226ab5cc92bcfb6f` |
| `insts_lfm3_pre_fa_fin_c16.bin` | `b5dbe71ea6c806bb88ecdd65b49c5f595d81f858d4c0f6a33341cd5bdc5484fe` |
| `insts_lfm3_pre_fa_fin_c192.bin` | `d686a90c1b2212e31702e6d5b1eaaf3c60aca310c015e051e11395a2f3ba52c1` |
| `insts_lfm3_pre_fa_fin_c2.bin` | `a4d1de9bdef85f2604d90bef2d7af41f4bc800fe878168b8d44e839b37a97264` |
| `insts_lfm3_pre_fa_fin_c24.bin` | `58b7613e68cad282d796e4c8396543b2de9e195fddd036da1d98db58fc6676d6` |
| `insts_lfm3_pre_fa_fin_c255.bin` | `c84adde5d9d3662a18f8fac5ff5a4e23103448e57ce8a5e3026527bcfbf5ac6f` |
| `insts_lfm3_pre_fa_fin_c32.bin` | `fa4eb5977fb17f5530f77030f984e50b336b61833573fd5532e76018b8977ce6` |
| `insts_lfm3_pre_fa_fin_c4.bin` | `03d598db233e89b4f1fb53ba150456c635a083a47022de1743f64fed228bcac4` |
| `insts_lfm3_pre_fa_fin_c48.bin` | `0e95cea891c6be54d359e598eecbc53000ca1e97e943952b89b2f362913a7179` |
| `insts_lfm3_pre_fa_fin_c64.bin` | `e065590aee6d0b78135a5131893e2244b7a8380ac0431ec99ef7bc664ce96be6` |
| `insts_lfm3_pre_fa_fin_c8.bin` | `9648292d6a0a0197574b11d47ae518a41bba5bdeb8efb37c9664f02402fdb696` |
| `insts_lfm3_pre_fa_fin_c96.bin` | `46f88cfc515c9d97ae11940e1a8c31e0e64a00b835176c1ccdf8cd721d979f33` |
| `insts_lfm3_pre_fa_part0_c256.bin` | `ab31830211bfe8ce90c0cecece5ee23e285184eb342b922e7bd8e4a2166e9bfc` |
| `insts_lfm3_pre_fa_part_c256.bin` | `02724213efc5faa6d51d2b0f71cae7f955a3a648d33b30038bbc8b9d26ff0371` |
| `insts_lfm3_pre_phase_a.bin` | `3f2d5428f0bda32b99917a47d2fcebc33a4b0c4810d835832ea8b8e111661a28` |
| `insts_q4head.bin` | `5f49c3a90f429c93cd79d27021d8ce1388e336a0cfd90fb875c0baf6186ec6d6` |
| `insts_q4layer.bin` | `6b485360000b24cf7a5e2e25a374786ead632b361537fce9ccb3b253ce75f238` |
| `insts_q4layer_c1.bin` | `29c159f33f7f48aa6af601c97b7c4aff03ca666f0bb4c115568fdad98f41bc93` |
| `insts_q4layer_c12.bin` | `927bf981bb849d3b88d33dac6fdd02c2498ad488ea4412e82fe474cdc905102d` |
| `insts_q4layer_c16.bin` | `1a85f7f4ad838ba993dfa3e377036bc31d39f0bdfb8305206f43f029fc4691b9` |
| `insts_q4layer_c2.bin` | `7d2321ed15c68360c1eae236fb6fa25a27638f405d8618ab1307f13453094a2e` |
| `insts_q4layer_c24.bin` | `c3599c5b957c3b7b6c9e260580cda640d64fd7b7c7594bd3ebaa48affd7ea056` |
| `insts_q4layer_c32.bin` | `abf2a37efab441b6d3d568a478f467b896b194f2a27e41af78820e5ac4c58d47` |
| `insts_q4layer_c4.bin` | `01c3c790a4bba7d6090adfbb513ffcef21ba421db4a0790424e17976a27db6f2` |
| `insts_q4layer_c48.bin` | `1713baa7792be4ab74bf6b7ad8787ea0af06cca43dc192ca89764d71f6a55bef` |
| `insts_q4layer_c8.bin` | `f948beb64d6b9902872d0015798245264fcd213e5ae6e0e80dbab27c7171065d` |
| `insts_q4prefill_layer_c0.bin` | `927d6233e7df9c9e42ff448f94ed4c353cd7866994d1ccf0699c3ee2c8e039e5` |
| `insts_q4prefill_layer_c1.bin` | `79dc8d2237b84309c00dd1bf153632199cdeec11c3d6097951a9771101d5eff9` |
| `insts_q4prefill_layer_c12.bin` | `b75899ef1440b5527e327e5453adf8e1e12c06df24bf5d4784dd06fcc9a58d0b` |
| `insts_q4prefill_layer_c16.bin` | `79ffd70ff616167248fca6ceae80adf5436173932f03ea5de4655b27d8688b23` |
| `insts_q4prefill_layer_c2.bin` | `17745257970ebe92bfe1da841730f3c520ceb51a089d121196c86570c092ed60` |
| `insts_q4prefill_layer_c24.bin` | `c4e43ce15acb4b333b08d094f4fa0ef6e379c1f62ab41d25b393edf50e211e66` |
| `insts_q4prefill_layer_c32.bin` | `27bfd978da96259806f2294f523f6b4c6fc44ca4227fc90e2f5037f9136b1578` |
| `insts_q4prefill_layer_c4.bin` | `a2100faaad51a2b0712b2a0c578e03440f5fd4f52ded7017dbdc53ade293f0f9` |
| `insts_q4prefill_layer_c48.bin` | `c5468b0d1884a98d15bee476e797c016ffda926ff3b1c3ce39848b18f50a8a64` |
| `insts_q4prefill_layer_c63.bin` | `4bcd89065f4d602239503d6c94b316734f7c40f6e5bc2abdacd85db7dd05f435` |
| `insts_q4prefill_layer_c8.bin` | `06f0fc7aa5ba11e0d88da3d6114a338ba8779ccd6559bc0ff226f70f75633fb5` |
| `q4head.xclbin` | `21613018d422e10c21072d7580041162b2658c053f371bbf7fafae8cdeddea05` |
| `q4layer.xclbin` | `21613018d422e10c21072d7580041162b2658c053f371bbf7fafae8cdeddea05` |
| `q4layer_lfm2.xclbin` | `3f730500c5a7c9742d8a252e0ca42f0267740730c9efba87e75b725367ae9410` |
| `q4layer_lfm3.xclbin` | `7771cf38da8e8affba51dfca356ec87ca4b924a9dd0d096b2dc33230b1f4b905` |
| `q4prefill_layer.xclbin` | `f5430e023ddc61af7b64aa2686a717a9ac7a21694ab3e352e617b2fa3bd24051` |
| `q4prefill_lfm2.xclbin` | `56c54db77c245fe4e4297e6300a2ed83ffac09da0ba57d4e6a828614f786938d` |
| `q4prefill_lfm2_ext.xclbin` | `b2f198c62dca1e68afd2f4b85ea7913b230fde289e1a55b3a72653f1b49f1693` |
| `q4prefill_lfm3.xclbin` | `66c3878b3574d155b167561b47a5249009e8a2bb5967155b6a1e7570df018faf` |

Weight and activation formats are the decision record's
(`docs/XDNA_PHASE0_RESULTS.md` section 4): signed int4 weights (Q4_0's
`q - 8`), one bf16 scale per 32-weight block per row, blocks-then-scales
tile layout, Q8_0-style int8 activations quantized on the core.
