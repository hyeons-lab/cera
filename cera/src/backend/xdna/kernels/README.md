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
| `insts_lfm3_dec_attn_c0.bin` | `fc2f5b80ed15286d55c34e157d0d782f60adb07c10108999093241994f173a98` |
| `insts_lfm3_dec_attn_c1.bin` | `4662b0df6002789a8fd4c7478a442ee12916055422eac0e63e00dd618f2901db` |
| `insts_lfm3_dec_attn_c12.bin` | `3d6d736be61655a062ac33ba46e67d0b48a64ffc8fc020eab3ef124d07b29397` |
| `insts_lfm3_dec_attn_c128.bin` | `f98be8020aad7e5620ecce33675cb47ca0f5eb94d536c175b55fc1679ae84a60` |
| `insts_lfm3_dec_attn_c16.bin` | `9e3d1ea1cbe2ab31299733dd18470f9c46bb44bc478815cb8367c25a420b4744` |
| `insts_lfm3_dec_attn_c192.bin` | `ba1b8c29d52ad3df3fddbe282387328c4df3bbea31fd09f48229838d544faf88` |
| `insts_lfm3_dec_attn_c2.bin` | `e968c68242f6b85ddc82e86956d3bb14c46423c794bb75330376c9c13eaca83a` |
| `insts_lfm3_dec_attn_c24.bin` | `ce15f06f5bbc1bc0b43b7ab83142aa4d867c5d59d3737208b60552c7bebc0737` |
| `insts_lfm3_dec_attn_c255.bin` | `f04c456b79bac9ac7f32fa3d42ba37069a78d64c07a9805d5fff70a15dac14d0` |
| `insts_lfm3_dec_attn_c32.bin` | `e6d440a34c65c39d9df32260410b78e0e0781e3c75dcd05d593dd6aa61c28b68` |
| `insts_lfm3_dec_attn_c4.bin` | `dd4d1fd8bce41f744bfd0825abe78d021d676427b100629d05a62cf36baf7b2e` |
| `insts_lfm3_dec_attn_c48.bin` | `3b0c004b97e0ff3147c4ef612f81123dbf03b6a19fa9c869ecebe0add7385cad` |
| `insts_lfm3_dec_attn_c64.bin` | `bf75aceb6783279d1427b65bb92fa8356959a4ef79f47e0f4ae50bfbccae0f21` |
| `insts_lfm3_dec_attn_c8.bin` | `c8bfee6b908d71e52aa116a1e2f191efede359d9a881dc71f3519cc7b733034d` |
| `insts_lfm3_dec_attn_c96.bin` | `19a2296ac99297d1d1cfedd0512d3a87907cbf419ec78d9d7801202a6bb95210` |
| `insts_lfm3_dec_conv.bin` | `4d349009aee20e4d1cb44320ff99b359667f9282b558490c3e7803e081f9ca57` |
| `insts_lfm3_dec_fa_fin_c0.bin` | `08c59d21dcbc58e024ca9508e8f6ffbb80153ec52804f625581e3fc8cd300a95` |
| `insts_lfm3_dec_fa_fin_c1.bin` | `aa9a2b62018a8f10834c85d535fe2a701c3ed38ac327c24cc64084e462967b66` |
| `insts_lfm3_dec_fa_fin_c12.bin` | `018c594c1b442a5e6f3d84fb9ca12ea8cf28fd32a5426d5f65edb542655c80e8` |
| `insts_lfm3_dec_fa_fin_c128.bin` | `9e32a8293e8332669b6a44b41bd3e6c20edef9228548f09fbd6f96365fb41fbf` |
| `insts_lfm3_dec_fa_fin_c16.bin` | `3bc85c7a983a474b0dc892080e32b6391b26d8edad9864a541f9e2c8260641d5` |
| `insts_lfm3_dec_fa_fin_c192.bin` | `c91bb99856fafd4bdffaf402f7c0274dc8c58b5ccb6236d9cb0f3f5d6c15d36e` |
| `insts_lfm3_dec_fa_fin_c2.bin` | `09c98deb5bc3a3747b9b51405c710fe737d3aab7aaa77285bb58efa726bf429d` |
| `insts_lfm3_dec_fa_fin_c24.bin` | `2b0028d33b1e35ee3cf46598629d8f19d5c00c2742d70f29cec478d45f4600d1` |
| `insts_lfm3_dec_fa_fin_c255.bin` | `480882c20d7949a24063e49c9b789049b61952667bbdf656ec02efea684041c5` |
| `insts_lfm3_dec_fa_fin_c32.bin` | `3a6988a983f8f9880aa29f87a1b00512c40bb3bd3a17ebd0f3aab3648eed2c7d` |
| `insts_lfm3_dec_fa_fin_c4.bin` | `35c902ead9d5d6c1d4849ace6de75123c877921855c6c1552f90865cfb2bbe6b` |
| `insts_lfm3_dec_fa_fin_c48.bin` | `967ef4c55b41ac5f6bf2efe8b753c393d9b44087b374fad05a3f40475646059e` |
| `insts_lfm3_dec_fa_fin_c64.bin` | `755f2b547c3dbd12a3f1a18aacd914875a2cad4f5670cc1e893e971e758bdc90` |
| `insts_lfm3_dec_fa_fin_c8.bin` | `f85715b76fcf785542dd5c3459b4c61c257e43fdadd70fe543cb25298f4913fb` |
| `insts_lfm3_dec_fa_fin_c96.bin` | `50bf908efaf0ef7900006b94f0dc8f893193ad4ede86bd80cbe699bffa1602dd` |
| `insts_lfm3_dec_fa_part_c256.bin` | `54cb6c416be9bd72e82a8d85be00d3624bea69ce2c39ba879e7dd70b605e079e` |
| `insts_lfm3_dec_phase_a.bin` | `92e4f34ddb884cb588ce61001b2d9d8a411cb3fd61ed7528447d0a7dc9211608` |
| `q4layer_lfm3.xclbin` | `bf53d07162e5f02def3411ba94ecd16fce866346ad5fa935892215870f0f702a` |
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
| `q4prefill_lfm2.xclbin` | `4d696b5ca32de3553df720315d961fdbbd0a71f89c3f04932c1aa62b39915314` |
| `insts_lfm2_dec_attn_c0.bin` | `65f8e3c91074396713522571dadddcbb7ade98f715d4dfd0503836a0761bfff0` |
| `insts_lfm2_dec_attn_c1.bin` | `66dd3629e7bbc22f3a91fc80bacbb263a6d3c8807927326e0f4f215fb1824988` |
| `insts_lfm2_dec_attn_c12.bin` | `96d7d9a2fe451ba980b0d0cdb8bba6df9cba38e8811e6f1a68ce61aa5e9db49c` |
| `insts_lfm2_dec_attn_c128.bin` | `7bd523b4b0262acaa28c42ba9336cc4fad3273eff9234f0ec2a67541445b2991` |
| `insts_lfm2_dec_attn_c16.bin` | `963ab27dc76342460317df3f90943fecba62e94076c459727601550c7a415b40` |
| `insts_lfm2_dec_attn_c192.bin` | `5506c36bcb72691bbb745172eae7fa6776a54f8da52855d40315a40e96ba8fa2` |
| `insts_lfm2_dec_attn_c2.bin` | `62212ea5cdc4363b4bee0061798fc6201cc7685efa6b05ec787da8f86af3d53d` |
| `insts_lfm2_dec_attn_c24.bin` | `a690fa2de73cc469dfb410cc3bee1db0f44bd4ac5c33ee47b9219de3b56d8189` |
| `insts_lfm2_dec_attn_c255.bin` | `236cf554541550cf944e43a5fb6b96700a4baa6d1bc688f5c280b374e1ab4f62` |
| `insts_lfm2_dec_attn_c32.bin` | `07bc4ba91c38ec6424974dd5ff00d9ceaa4b005db91671f08274a002237f755a` |
| `insts_lfm2_dec_attn_c4.bin` | `eafb335c471b86aa7a33d27a88b29ff3db96c03b4d032ba12506e8c53382b85a` |
| `insts_lfm2_dec_attn_c48.bin` | `1215c2dde42afbdd2e1bb21f11934d14ee08487fa7c24922091904877fe0e4fe` |
| `insts_lfm2_dec_attn_c64.bin` | `fafcf6454e698b55e0d72ca4ce0a4b5d8e4331c2d2be106cc1d5280c8506f4b2` |
| `insts_lfm2_dec_attn_c8.bin` | `6d12f682580e828ac2c182c5fd8c391d9534aadc4b43f87538be232c35fd76f7` |
| `insts_lfm2_dec_attn_c96.bin` | `fe07893e596a2c98f1efd026b815cb68197b2af63a3fa3ed6fb8aae685810d1c` |
| `insts_lfm2_dec_conv.bin` | `d595337b4064942cd8d2fa3eaf4432ab50de05a0e349ca283d913f2119aa5b20` |
| `insts_lfm2_dec_fa_fin_c0.bin` | `d2c3c93bc43d3f65742395a7afbdb9c6063c06127934f50be13e3ed907249be0` |
| `insts_lfm2_dec_fa_fin_c1.bin` | `5794a75f4dfb2169b0f6ac72c9a7f91bd0b304630fcdd16fcf43a096daca06c3` |
| `insts_lfm2_dec_fa_fin_c12.bin` | `e409fffe9306391d44806af678c12d539bcdc742e2e69df6eaba8013fa7aba88` |
| `insts_lfm2_dec_fa_fin_c128.bin` | `549c42103d633ec1173a73ae8f6c26f13411dd554fef15f66f4c85201761faca` |
| `insts_lfm2_dec_fa_fin_c16.bin` | `430a44757623ac330673e1c50ca5ac3e96a39121c2063900125b29444376c646` |
| `insts_lfm2_dec_fa_fin_c192.bin` | `55179ddfed1c71a96bc448f9b2f563efba54426b3ec53f8e2c60b3d680564a20` |
| `insts_lfm2_dec_fa_fin_c2.bin` | `91a02bb86e40d64023df09e188d9d3810ed903016d65fb783809e6d61aa48d68` |
| `insts_lfm2_dec_fa_fin_c24.bin` | `9edea51929a1069ffc6f61a5810477043959cbda73728e85aad07f34812a7335` |
| `insts_lfm2_dec_fa_fin_c255.bin` | `00afdf3e4ebcd6b2281ab31d4aca8d2e4e121ac7fb78c883f1b1c3c8841c4484` |
| `insts_lfm2_dec_fa_fin_c32.bin` | `701df6e11e0d286e31ddf470da73ea0f33305eb08720867926e2a803089b2f38` |
| `insts_lfm2_dec_fa_fin_c4.bin` | `39550a181509e91b3a522e08e518f61c829fad6fd264a6df46bf9ab92a01810d` |
| `insts_lfm2_dec_fa_fin_c48.bin` | `79c34720dec7501ab0a867dafd9c3e22883bd5c6e2adcd66a6b56c4a5d7b35de` |
| `insts_lfm2_dec_fa_fin_c64.bin` | `d0aefe08ecdd9f44d9bcd16ce4f31d62582575f5e4d7f5b6c6b9b0bc7e645ff0` |
| `insts_lfm2_dec_fa_fin_c8.bin` | `bb7f4773fe8dcae6bfac484ab5be4ba9d7dc98b524ed0f60609065e46107d157` |
| `insts_lfm2_dec_fa_fin_c96.bin` | `29469b891f0170dfcdfe838f24584f645f297431d90507f3484af2c897b60125` |
| `insts_lfm2_dec_fa_part_c256.bin` | `696f518930e9a9eeeced15f85a73323d951fafcd3e8853e0572c9e8bb51ed733` |
| `insts_lfm2_dec_phase_a.bin` | `05527be542f4a5f3a07dff2640a140103b39127331e07196a09748f08fe59579` |
| `q4layer_lfm2.xclbin` | `fdb205c8da85221373d60dab69bf2b688f811844277800e3e9c21d67437e5281` |
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
