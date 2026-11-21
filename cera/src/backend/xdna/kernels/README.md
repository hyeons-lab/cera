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
  (`insts_lfm3_dec_attn_c0..c255.bin`), and >8K paged Modes 2–4
  (`insts_lfm3_dec_phase_a.bin`, `insts_lfm3_dec_fa_part_c256.bin`,
  `insts_lfm3_dec_fa_fin_c0..c255.bin`) sharing `q4layer_lfm3.xclbin`.
- `q4prefill_lfm2` / `q4layer_lfm2`: unified LFM2/LFM2.5 prefill (`B=64`) and
  decode (`B=1`) kernels covering ShortConv, SWA/causal attention, and >8K
  paged online-softmax modes (`q4prefill_lfm2.xclbin` + `insts_lfm2_pre_*.bin`,
  `q4layer_lfm2.xclbin` + `insts_lfm2_dec_*.bin`; legacy `B=32` sources remain
  in `iron/`).
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
| `q4layer.xclbin` | `21613018d422e10c21072d7580041162b2658c053f371bbf7fafae8cdeddea05` |
| `insts_q4head.bin` | `5f49c3a90f429c93cd79d27021d8ce1388e336a0cfd90fb875c0baf6186ec6d6` |
| `q4head.xclbin` | `21613018d422e10c21072d7580041162b2658c053f371bbf7fafae8cdeddea05` |
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
| `q4prefill_layer.xclbin` | `f5430e023ddc61af7b64aa2686a717a9ac7a21694ab3e352e617b2fa3bd24051` |
| `insts_lfm3_pre_attn_c0.bin` | `65a19cc5734e7e5fa189850294b239f73a587270246d5cf9b8dd60d1dcd51810` |
| `insts_lfm3_pre_attn_c1.bin` | `9e44b76400636f4fca0f21d65cdca1d164a8369aacd638e112567ff9165fa379` |
| `insts_lfm3_pre_attn_c12.bin` | `b906be4836bbe4c8be0a2c8b294fb6d44c0b87c843807e7a2a466a3654c1598f` |
| `insts_lfm3_pre_attn_c128.bin` | `1289628dd84e934ebd33a3daf5f57144702f2d1b5755146ddce09d6404df1537` |
| `insts_lfm3_pre_attn_c16.bin` | `7f819f2867b88b7b300e07370615a2924e3ea47df171d48f97f13cc74e4c1015` |
| `insts_lfm3_pre_attn_c192.bin` | `04dd87bf5abb41ddae1b7350bb7ce1b4c96feaffabed4f0d3288ddea76672a33` |
| `insts_lfm3_pre_attn_c2.bin` | `f678aee8e070c7b2830a4ab138bcba546a0e803a8bfebc9c19be9aff928daf29` |
| `insts_lfm3_pre_attn_c24.bin` | `04d9a432d4a1b0a4f4839e2bc91e4d089f770c71c3e56cc062e8fdb11ef4765d` |
| `insts_lfm3_pre_attn_c255.bin` | `fe3db7f31c9ef7cd2a8eb56e28268ae2f73ffa768f49a3f06a87a2a0506c55c3` |
| `insts_lfm3_pre_attn_c32.bin` | `897e6f432edd0ca2c91433c7fb39a54a357391362a06800657721d8f55c208a4` |
| `insts_lfm3_pre_attn_c4.bin` | `4d8786dfc9cde4693772fd9988fd21b130ad168d6755cb1463235b35b0bb2000` |
| `insts_lfm3_pre_attn_c48.bin` | `e52c7ce0582210c1bf38998d714051352c6e311a7b724cd53e6fd0c24861bd63` |
| `insts_lfm3_pre_attn_c64.bin` | `a8e745fe5ad12f712aa4e3b89fccba2ca5c07d90c8a8baa9fda3a7b96c4915fd` |
| `insts_lfm3_pre_attn_c8.bin` | `db3250ae251f29f27afcb02b2fc7f2233e85b8a3e562141527b93513d9e488c4` |
| `insts_lfm3_pre_attn_c96.bin` | `2a96562da0b4c272356c8f329713932bfe2d6240610d56fa9f5563d00a19d529` |
| `insts_lfm3_pre_bidir_b_c1.bin` | `cd1e4f23f403f780c4dd0e84a672dc342cc1d4eea25405aa1db5c583aa30c54f` |
| `insts_lfm3_pre_bidir_b_c12.bin` | `e382cfa84b24dc92597c094e3c119f2e57adda5a47fb19d46a51589306e6872d` |
| `insts_lfm3_pre_bidir_b_c128.bin` | `aa923c1476829a145fa83837cdca104a01c47b771a6bf13b8ac44aa79b1b9938` |
| `insts_lfm3_pre_bidir_b_c16.bin` | `402bfcf02746cd57ae7c03854aefb90779c92b0570bb182caedb9ee876092d20` |
| `insts_lfm3_pre_bidir_b_c192.bin` | `4333903673e0ba51eb319a3de671e342df988199bb6f53e7592b24ade8dfe38f` |
| `insts_lfm3_pre_bidir_b_c2.bin` | `10b7d4be90b791fc4aa23f7d190ce25ca7a857da357cedfefb6dabed6dd82c44` |
| `insts_lfm3_pre_bidir_b_c24.bin` | `0361b12fbe1d555778c6b7499b54cc31944a429e913f8fa027d789a18b3c91b6` |
| `insts_lfm3_pre_bidir_b_c255.bin` | `be8f3d04d486e7f87c2dd9c6c9ba9fd0ce8b48276a059f0ac0220a95716dc067` |
| `insts_lfm3_pre_bidir_b_c256.bin` | `aa984006feda640eb2c533e773033cbabb15682fa31dc2f6b2ae84c375910d18` |
| `insts_lfm3_pre_bidir_b_c32.bin` | `e2320dc72827614acd403a27b02a7dd10934fecfa96b161e72720384a90be314` |
| `insts_lfm3_pre_bidir_b_c4.bin` | `4dc93aa8d585a912e83ccbac634ec7a9b83503e768f5755f2047619c83968153` |
| `insts_lfm3_pre_bidir_b_c48.bin` | `6147381b74fd884142ee5c8108277dcb0c8c3a6446e2c392c75629d5c2c98439` |
| `insts_lfm3_pre_bidir_b_c64.bin` | `e44e5efdfe835212c36cb63511d36009ee8cb113740995e7fa8e5da3b6e22257` |
| `insts_lfm3_pre_bidir_b_c8.bin` | `20870a5b50fae3a3f1d05d762268581671c0bd117747751df8c66d2f0144e7c9` |
| `insts_lfm3_pre_bidir_b_c96.bin` | `4d9456c1988d75d8e0c82aa1eda1c0dea56bf327d1f9681db4acb7a64b24470d` |
| `insts_lfm3_pre_conv.bin` | `80429c03f40f82437bfd78b4e64cfbe47ce2261e54490781d78533cf1864506e` |
| `insts_lfm3_pre_fa_fin_c0.bin` | `c1fb4b3c1aa44efdcb3f07a76d8257fdc5a7c01b8ef1721b5e84f43aa04ca823` |
| `insts_lfm3_pre_fa_fin_c1.bin` | `4767a7d0d53c041e45d3730e8880b30ef3d64735fe3a009d49fdf059cc565dc7` |
| `insts_lfm3_pre_fa_fin_c12.bin` | `0be408293b2108ea998a14bf75ff467ce7a033d1158ac20c95eca161a2a772c6` |
| `insts_lfm3_pre_fa_fin_c128.bin` | `74a56b56a0fa422f35e495e5e83998a9f97e5612760ead6285caa58680b183a0` |
| `insts_lfm3_pre_fa_fin_c16.bin` | `a5c71a8bf885c1fb557515efdbee989f6d84a1000716f0415167a2bd9875bd33` |
| `insts_lfm3_pre_fa_fin_c192.bin` | `1c769555db1b1dfca2a89d3c99fe1239b4101926603b65b41fb44019e14bf9d3` |
| `insts_lfm3_pre_fa_fin_c2.bin` | `a457c5f7c2372d0b92e29d104d6a51f30aa5db1f2078218755177b417eb68e9e` |
| `insts_lfm3_pre_fa_fin_c24.bin` | `794e69f7a6907ddcbeb98b2f10ce3a1ed09cd0bb53d0619474431df24023f270` |
| `insts_lfm3_pre_fa_fin_c255.bin` | `6ac5494596f987cdb29c252ab60529a044e63c193ead0b6b495d1a9ed31f96b1` |
| `insts_lfm3_pre_fa_fin_c32.bin` | `c0bbf3ec841f547de74d05f98ff2cb5c56a6f5bfd50df65fd90709cdbef23f37` |
| `insts_lfm3_pre_fa_fin_c4.bin` | `448964c2140a5c601b2fa8805d69ba59575eeac85ce9e24b9dc1f989f5ce51d9` |
| `insts_lfm3_pre_fa_fin_c48.bin` | `11cb3c672ce4159b969c99b1da13d2f51a539e103b9ad0c01a95d117d8dfbc82` |
| `insts_lfm3_pre_fa_fin_c64.bin` | `540f64220e2a4afa95444740e0d074777f2282bf2a739b9e033c581938dc5d0c` |
| `insts_lfm3_pre_fa_fin_c8.bin` | `69d8901f073230038e2ee3bd64f965c38fa2ad0acc0df706c23d7ced465a4739` |
| `insts_lfm3_pre_fa_fin_c96.bin` | `c023df835aee2d1c6764fe94a23b11122dfa8ce18d20ac02f9e2a21ab2e4df00` |
| `insts_lfm3_pre_fa_part0_c256.bin` | `6b7efee7a041923b9e9d09ae125004e49e93ff614248d75323e65a077988277f` |
| `insts_lfm3_pre_fa_part_c256.bin` | `c5068419dd29fc0d5d950f694e8d311cee4c8697f63291f2994d45757fe89aea` |
| `insts_lfm3_pre_phase_a.bin` | `17fb68860f911723f557211c9f3d6e094f4568e5d80227f862fb1f3e63fa58e0` |
| `q4prefill_lfm3.xclbin` | `21344f8f72364f8a6d4c55f1d4ecee1828755f03ba6055509b425fdd5097b36c` |
| `insts_lfm3_dec_attn_c0.bin` | `121fb12c92e3e963b64e3e4381db1c9c5a32cc74e9a22034fcd15e885502cbb2` |
| `insts_lfm3_dec_attn_c1.bin` | `d8b339c93e1de7905dcc2a99b94f5970703fcc4490e4a30a1bc5eb7a8a5246d6` |
| `insts_lfm3_dec_attn_c12.bin` | `0c901dc20aa9d33248231bfda489c3abb645f60254f64991ba9abbdc0db3f55b` |
| `insts_lfm3_dec_attn_c128.bin` | `5486cda0ac099c7dec701ca1b5aa5dc31f9280e5b91e7e2518ba2c44e829ff98` |
| `insts_lfm3_dec_attn_c16.bin` | `26a16dc5a9f726d4b5661899455a5b68ff70945f3db2e239f4e5593927962a83` |
| `insts_lfm3_dec_attn_c192.bin` | `9f14547c4ea31b32492757a2e5e5eb9349e0bbed3efec599469dcca975eceb12` |
| `insts_lfm3_dec_attn_c2.bin` | `a232861e2d68cf5a0d57dbc5ea2820693e9df74b749f2e00368c5957f364169b` |
| `insts_lfm3_dec_attn_c24.bin` | `772b09e22e426ac07e1fb2429de77b27a0d42a164133311bc6ddd8f37780d4d1` |
| `insts_lfm3_dec_attn_c255.bin` | `780e437610d7f167e66a06d70d54f556e5c916c80849b48705b45f82bf32834a` |
| `insts_lfm3_dec_attn_c32.bin` | `e9999266fcf9563cd8bf5fd23f5b9cc7e6c5da82698a5ca30e04103e4018fadd` |
| `insts_lfm3_dec_attn_c4.bin` | `f53480b5dde45b8ae51d608927c71799bc46050519c00a4c33e54fdc2b5ebb89` |
| `insts_lfm3_dec_attn_c48.bin` | `c7d366f8eba9a0ad1bc0fb0c3e8fa7b3850c7edf5ae23a117bd184fa0d4c65a8` |
| `insts_lfm3_dec_attn_c64.bin` | `7b7292748be66d1e5477568cb500900df3d81395c702ec7703b88f87d5bf8446` |
| `insts_lfm3_dec_attn_c8.bin` | `23061a72c3b6acaee5a9d953a5cd284c5b769f3da9a1c9e75bf9cb70b5d6c6e1` |
| `insts_lfm3_dec_attn_c96.bin` | `187b4b7d5479f4ca0e5e28f3466fffc18fba999fddc36e0791b728fc40ff858c` |
| `insts_lfm3_dec_conv.bin` | `a010677024534a5c9e6051ae257e2a032c9f4794e86218c58b5ae53cd41bd542` |
| `insts_lfm3_dec_fa_fin_c0.bin` | `596c3b4b5c3a598c055adb9e7a3d5b513bda2324aa89390502add694d0adb9dd` |
| `insts_lfm3_dec_fa_fin_c1.bin` | `70bf838ce6097f38d29c165474b7426df446290465bd29de3c0848c65de7344c` |
| `insts_lfm3_dec_fa_fin_c12.bin` | `ccec1c7b2b5769142767ba172937d9aa49d8c18a736ab6a1b7d0746b9307dc12` |
| `insts_lfm3_dec_fa_fin_c128.bin` | `026a55beefa717f1eb87a5375a4972e4f065bf121cf1a881b01f82a5ea77671d` |
| `insts_lfm3_dec_fa_fin_c16.bin` | `549f5ff9ebbc6744ef77e3ca2c384f45bb1d4235c078f68ffcdbe4774e9bb8ff` |
| `insts_lfm3_dec_fa_fin_c192.bin` | `4cf6f5d85f4dda913bc5a920d14804d22eef8b049df9b723c56b531bd9c152db` |
| `insts_lfm3_dec_fa_fin_c2.bin` | `1297219722d0c53bd8bea0865289738c8c6adbc75a8eb5684f5769a31d89c1f1` |
| `insts_lfm3_dec_fa_fin_c24.bin` | `c42792e07d07441d914d40d42c8639adab36775333fca09c27bf9dd5c3211576` |
| `insts_lfm3_dec_fa_fin_c255.bin` | `e4becdd3dff8ab338066a2522a586356ccf2723bd99b7b2ab6f9dfd64a05e9e9` |
| `insts_lfm3_dec_fa_fin_c32.bin` | `d919ab3eec7e7aeeaaddec0af68d2ddaa64478ad2d6cef7feb82dbfd6d774476` |
| `insts_lfm3_dec_fa_fin_c4.bin` | `2c9fd2faac147f835175cc43178819613e42dda56a719858c95d279f14222b6f` |
| `insts_lfm3_dec_fa_fin_c48.bin` | `3aa34014561dc6b3e306fb001a829c682bae5906a0a009688a038e15c69d7599` |
| `insts_lfm3_dec_fa_fin_c64.bin` | `bc09f24d39ec16a445f49f50eebf5e97325933dcae18e1456caa20c9424a10c3` |
| `insts_lfm3_dec_fa_fin_c8.bin` | `45e06ab1e2a5bc42f949ea518eb1343442a48eeaa49f5be96c59ebd6a3254881` |
| `insts_lfm3_dec_fa_fin_c96.bin` | `b5cd7cba6adf6b436c7a578f4bd5752758fd21533ed05c4778487891c4559242` |
| `insts_lfm3_dec_fa_part_c256.bin` | `f0963e4a938fc791aac782a16eba7c049282011ceaafcb600c587360b7fcc578` |
| `insts_lfm3_dec_phase_a.bin` | `630a1f75d90c160e10bcd91e776039c51fc625a59b40619946e46e85be7c307b` |
| `q4layer_lfm3.xclbin` | `d2cb5e7bc00ee142c863ff5429f3cfb89d4ed63620d2e14be0b46b22e9cdac21` |
| `insts_lfm2_pre_attn_c0.bin` | `aeb8a73662a15c8aeffcf1411d123d1690eab684398316cbdc97c353babb8640` |
| `insts_lfm2_pre_attn_c1.bin` | `a0c700acb2d071aa70f5a6906daefde2e4ba400b606744b9126ec5f8feb04208` |
| `insts_lfm2_pre_attn_c12.bin` | `1dfab33454d11881afceefd95d51b292048f93ccd370352586c991681c09b7dc` |
| `insts_lfm2_pre_attn_c128.bin` | `9d714ae0066e62b5870a3d6923b9f5734ddbbfc00e0fe89767c85a508579422f` |
| `insts_lfm2_pre_attn_c16.bin` | `69cabd1ad81f7693f866ea54f3b16e4fbbb05764e5f17e9947b42c531d46ee57` |
| `insts_lfm2_pre_attn_c192.bin` | `26b5ff5a8d2d5028ec18110501ccc5a99d82640bfb5473aa6a2ee389e6ae936e` |
| `insts_lfm2_pre_attn_c2.bin` | `edfcb068baf51d04d53300373ad33a64a4f60e6186cabd3568da8a669183d100` |
| `insts_lfm2_pre_attn_c24.bin` | `2460e78ff0d561ed9d5b66484c932eb7523f62c2133f0869a9d66ff711bf43dd` |
| `insts_lfm2_pre_attn_c255.bin` | `873b87f64cdc584fc43832ef6aad1feebf92f6805ff18314cd4795525fb94649` |
| `insts_lfm2_pre_attn_c32.bin` | `54a85f88e68018cf46393d8bb3a389fb7be9e18e7f2ec1cc541ded0615441dc4` |
| `insts_lfm2_pre_attn_c4.bin` | `e139ff0704a71e71e0138978b3a55888902acca16d2ddd7639bcdfa2e1d5f434` |
| `insts_lfm2_pre_attn_c48.bin` | `1cf37fc9cdac1f8af1e8535530d715b4026d9194d22cc206164b74349a4304dc` |
| `insts_lfm2_pre_attn_c64.bin` | `a9ea1bb0df24e8f7bae1aa24789b519f23481b3aa0529b642b53489e5c7bd254` |
| `insts_lfm2_pre_attn_c8.bin` | `858b218f8dcac242ced9e542a419f6d0404fd22b6d5b88bc9fec81ff1e285723` |
| `insts_lfm2_pre_attn_c96.bin` | `db4cb1ca6b0f44be29bb895dba5b73ea53bd506f7c6b6599e212b75e2d600e85` |
| `insts_lfm2_pre_conv.bin` | `673dba1f904203120e93447005008361a84a6c6a41f76d6fd0a83fc6379a6899` |
| `insts_lfm2_pre_fa_fin_c0.bin` | `c251a6f3972c0ee1cb62ebdc594d2c6ed3eb953cc2a00dcb7700d3f2b215c0a9` |
| `insts_lfm2_pre_fa_fin_c1.bin` | `d741fad6d67aee5b4c4422eefe7388bd9ee5fc8faa7d8d5a7e439d8e6a517940` |
| `insts_lfm2_pre_fa_fin_c12.bin` | `054449252a39417a422cf0c4d07ffc457a83ea94da1b8fd61c8a6544358a304d` |
| `insts_lfm2_pre_fa_fin_c128.bin` | `7d96c6887ae18dcef951f9271ec0806bd851e6f27688415f87c3a6cfdab7b032` |
| `insts_lfm2_pre_fa_fin_c16.bin` | `6ae90822d981a26b101e26176aecfc0494c4c3dad077093d9bc492b36dfff79c` |
| `insts_lfm2_pre_fa_fin_c192.bin` | `97f8ef91e88bd07690d40977f458cd65cf169f7ab2b014fd0d8e00315b50e949` |
| `insts_lfm2_pre_fa_fin_c2.bin` | `27362aada22d7a00ac6c1e52c842f7f0056b7a8eec9fa5b2c33c75fdd06cb11c` |
| `insts_lfm2_pre_fa_fin_c24.bin` | `be1c3167ba9e39e6f388d654fb7e9ab120cb22939eb0b394eecdfa8f587a1d60` |
| `insts_lfm2_pre_fa_fin_c255.bin` | `bf895c042b3626ffa5afe68b8310ed565db0b0765818de28891590ac9ee4a4a6` |
| `insts_lfm2_pre_fa_fin_c32.bin` | `e22e6a02014f9662bf66752b01510d3dc6e58a4a6911d7926625a4e96b1fe9ec` |
| `insts_lfm2_pre_fa_fin_c4.bin` | `c6f109fe9d4aeb89a3c99ccff2269d4c1a27c9bda27780755ae830e0623256f4` |
| `insts_lfm2_pre_fa_fin_c48.bin` | `7af76b0ea52ca4ae143d441b12356926d02643147a7f6d5eb94385727b20d96b` |
| `insts_lfm2_pre_fa_fin_c64.bin` | `d6bba47d8623f075b1632096fe19f4446d6832c9e8a32b606e822b44998ebfc1` |
| `insts_lfm2_pre_fa_fin_c8.bin` | `b3096d50e64711bce25774c80c3d1b690912710886a06af99a6b978d5eacec7d` |
| `insts_lfm2_pre_fa_fin_c96.bin` | `0cb7bb2fe70d1da9884453df75899cc339e60593c7a42078a55e43783a2dc84c` |
| `insts_lfm2_pre_fa_part_c256.bin` | `3be6c1be4618bda6d14cef3a2b88649bde9bf5a8eb2a0010a10b557e1aa5becd` |
| `insts_lfm2_pre_phase_a.bin` | `1fca97ce39141b2533794c48cf02553907166d98a7dd5a99252524f917f0232f` |
| `q4prefill_lfm2.xclbin` | `65276157dd2659dc282a16e0caf6a25ebdb632d3d0ecc62b092ede43e618fc47` |
| `insts_lfm2_dec_attn_c0.bin` | `2407e0c5d2ca05845630e92131a2e253b31e3128f0e00827082c86905df51863` |
| `insts_lfm2_dec_attn_c1.bin` | `fe4772411ad0db7ee89465863e18d38344a06f7609501cc6f9df57307693d2e3` |
| `insts_lfm2_dec_attn_c12.bin` | `00469f22df6c5666113d6306e182c59a4ff57763c03f2c4e01b6b5c42692ecb2` |
| `insts_lfm2_dec_attn_c128.bin` | `82a6a26eca24eb2c370fc9c17aec65e6a38d987939c42c0daf8d92b3c57b34c5` |
| `insts_lfm2_dec_attn_c16.bin` | `ae47e9cefbf64a547dd02d38c64d677da7abad408abf8c5592b9c74694d31138` |
| `insts_lfm2_dec_attn_c192.bin` | `fceaf2d920855ca723478b14601d44de5716f762fb612d88fef35f70f30f57b2` |
| `insts_lfm2_dec_attn_c2.bin` | `fe00d5e488a32dd863fc4ef2e579f92d95cd7ab9488dc74757745502a2c2649e` |
| `insts_lfm2_dec_attn_c24.bin` | `e63fff179dd3cd3ce9bfc160bfe4ef1abb5fbec4b0197ada37a08b062b5142bc` |
| `insts_lfm2_dec_attn_c255.bin` | `05c2c5e1f70fd93f73400b5aa6e1daa087e893fe8f8e15a0f925d3e1d7b90c51` |
| `insts_lfm2_dec_attn_c32.bin` | `6c10bbfe703ba9cb10c4f5bdc55cd9af82407bbed2b7e1c6b62d4d921cd4d3d4` |
| `insts_lfm2_dec_attn_c4.bin` | `c768da570dd170ef0f7737a0bf124948ca6a261f71e50507879b622566682389` |
| `insts_lfm2_dec_attn_c48.bin` | `6ac2102ffc5258140b4e7d27feac1a33e0291528a7c08f3dccf7731c33a6f8be` |
| `insts_lfm2_dec_attn_c64.bin` | `a8563cdf78735e42836b8c1c0fc23b58115ffebf5118db2ea045ac3aa0cb1ee6` |
| `insts_lfm2_dec_attn_c8.bin` | `7afc6a1ab75510f9ee8e2b208855aa9b1fd1f7ecbd57094773445c4137475c93` |
| `insts_lfm2_dec_attn_c96.bin` | `9cd54ff7944216f16dba7f13ebc93d0a3e01336c65403603e511807ca0c22771` |
| `insts_lfm2_dec_conv.bin` | `38f958b18346a033d66559db104219a9262f305d18096335ac9b1ef77120fae8` |
| `insts_lfm2_dec_fa_fin_c0.bin` | `294f22f9e87f273738eef22dc724a335832e105e47493e7cea9b6ed0671b5c6f` |
| `insts_lfm2_dec_fa_fin_c1.bin` | `000ecc295f394e374162bcd6642d95401d4b4db2ea5664c98b0ab3f7e7ac3e1f` |
| `insts_lfm2_dec_fa_fin_c12.bin` | `b001a8ab650df844f3996920190ef81766c1036eeb2fc29b9aaa8f5e4cb2c40e` |
| `insts_lfm2_dec_fa_fin_c128.bin` | `14f159573a0c9a40ab84a7bb33bf8088c10305e06136dc6ea8aa28ceb1dff3cd` |
| `insts_lfm2_dec_fa_fin_c16.bin` | `b9bce413d1a212524533cb271419a499afc84a7925d4651a8ddfc5da0ca798a0` |
| `insts_lfm2_dec_fa_fin_c192.bin` | `65e2e8d234c4d6ee26be0472d59f43defd01996092da9224f5772eeaf78242e2` |
| `insts_lfm2_dec_fa_fin_c2.bin` | `4c180992d4929b399a406901319b72b8403d1536ac7565249ac72a41f4198840` |
| `insts_lfm2_dec_fa_fin_c24.bin` | `34e8af30628693f6f2314e6a5669661c6918ce722b56c788126d32de496214e9` |
| `insts_lfm2_dec_fa_fin_c255.bin` | `ad311ac717b480c75b93e51f9213748517c472da10b2acd90efaa3aafcf1c0c9` |
| `insts_lfm2_dec_fa_fin_c32.bin` | `844b06e3f2d0f782e85303e2ff60a293910fd80b30fb779d6cf90c1cc02260d0` |
| `insts_lfm2_dec_fa_fin_c4.bin` | `b5c779773e34d7ae8247fd6c38cb3c2a7c23f230e3f5526d413363537e0a309c` |
| `insts_lfm2_dec_fa_fin_c48.bin` | `6c37fb118b23ec8005a39e07a02fc23a97913342fc2db8aa3ea7ed73767556ab` |
| `insts_lfm2_dec_fa_fin_c64.bin` | `0bc476831830b755b297db0cf9a45b876b9b57b1bc5d55283da03e16c6898dea` |
| `insts_lfm2_dec_fa_fin_c8.bin` | `87d43f4b0913b7d6226cd3c1338abd98db925bc6efde7f0f810abceaa703862c` |
| `insts_lfm2_dec_fa_fin_c96.bin` | `89024eab6368f7f2414fd757814578a5a8aa1470b479aeb06b80459f113fd6b4` |
| `insts_lfm2_dec_fa_part_c256.bin` | `45b3a6abe2a1e665207c870ebc730462081d6113fc8555666f6c3ccc5e810d85` |
| `insts_lfm2_dec_phase_a.bin` | `5eb518352b4bb5f5369816803e2580d91ff509a70a076e2545708ad1a0164f58` |
| `q4layer_lfm2.xclbin` | `290fc050dd76efc6071c2e118e776b5a6a6d46374179f80529727781f4dcfbd0` |
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

Weight and activation formats are the decision record's
(`docs/XDNA_PHASE0_RESULTS.md` section 4): signed int4 weights (Q4_0's
`q - 8`), one bf16 scale per 32-weight block per row, blocks-then-scales
tile layout, Q8_0-style int8 activations quantized on the core.
