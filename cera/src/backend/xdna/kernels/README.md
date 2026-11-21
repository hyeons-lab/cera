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
| `insts_lfm3_dec_attn_c0.bin` | `9c73ce435e1d90f2208d65e0a11bd7a298d3767cdcf4a50fb3bba6810e9044b0` |
| `insts_lfm3_dec_attn_c1.bin` | `e4771163594e6565b8cf28fbd35b470ae7bff15b1778e9842539129d105f46e5` |
| `insts_lfm3_dec_attn_c12.bin` | `093405d6e6ef1e0ce8477bcc7351bef23f85301462c996e6df268cb74e077883` |
| `insts_lfm3_dec_attn_c128.bin` | `8eb1865d7a74258fbbcd45851f941fde457d79cde1ca14ba0fb5f96f8cfa365b` |
| `insts_lfm3_dec_attn_c16.bin` | `078e9352ca7df564f1816755155df2ee9bf5ec423a457ff1768e29deaba3784e` |
| `insts_lfm3_dec_attn_c192.bin` | `2af03116f08d81b10091e6fa1ed938f2c095410d16d7ca5494888a8b270d8d7d` |
| `insts_lfm3_dec_attn_c2.bin` | `60c80148d694886c296858ec692662b0144dfea3c8ef55090563a35727ea7e92` |
| `insts_lfm3_dec_attn_c24.bin` | `c97cccbbab33a9a2a4bbe11271277dd19f3995530ba348e81211824146f599e6` |
| `insts_lfm3_dec_attn_c255.bin` | `dbf361635ae9ea3b204148666558cec8c957c9414e4458cf223cd6ce509098d2` |
| `insts_lfm3_dec_attn_c32.bin` | `f63fad6cd7d6560da27df6371ffb4c0b0e987dba759ef9f9b59824c73141e49f` |
| `insts_lfm3_dec_attn_c4.bin` | `5b0abcc6b68fba2b8d397435228e58ad0d96ba85f2ff39c47d0e2a513632ad68` |
| `insts_lfm3_dec_attn_c48.bin` | `faf24e633abb684f357d5b48cb3f7027d0485db7a1b9b1591e3fa887c3f32237` |
| `insts_lfm3_dec_attn_c64.bin` | `40fe8674dfb1736620465c86e3664676b02dedcb56a45323fa895420e68d0bed` |
| `insts_lfm3_dec_attn_c8.bin` | `d7568ac7a8d85fa2f4a556aa0b05bc52f72461a69b536f4a4db1d9850025921c` |
| `insts_lfm3_dec_attn_c96.bin` | `5cca3617a3c4659476ed830210b05e683e06add8e23eecea1927423980da8463` |
| `insts_lfm3_dec_conv.bin` | `b435890b935c7b7a4faf75322ad8b26672fdd3fbc5da35d656d31984af0f0a59` |
| `insts_lfm3_dec_fa_fin_c0.bin` | `5e99ef5673a992d19b07d4af1f4363fea184ec704d9fb0bda5c3a998d831061c` |
| `insts_lfm3_dec_fa_fin_c1.bin` | `687b7c1d17594960dd713262b0b31f7fafeb1812163405762f3a18454c875d62` |
| `insts_lfm3_dec_fa_fin_c12.bin` | `aa90e37fa4069bb1f5642f71d0f9547175b175a53136ca4eb855058c7dff610c` |
| `insts_lfm3_dec_fa_fin_c128.bin` | `9410a50132c9febc2671b1616ca9f809e03f05503babd7a0c4f159abbcbb7464` |
| `insts_lfm3_dec_fa_fin_c16.bin` | `0f4f9e4e6d444bb7f2eb6dba98f23741800594410e97fe23f099d317dfb0128e` |
| `insts_lfm3_dec_fa_fin_c192.bin` | `2fb41431ca60568af182a7b783c79fc473036ac4d5e001d11bee584ce420bd86` |
| `insts_lfm3_dec_fa_fin_c2.bin` | `b0dfe2078be671c0a8fb0ddccc5572c377aeda638b0bbbc491234585f5554836` |
| `insts_lfm3_dec_fa_fin_c24.bin` | `99eab1d00bcdac34dd1d9b04e2bdda18a2a91d41154c7531485e7aed57dcbb53` |
| `insts_lfm3_dec_fa_fin_c255.bin` | `cd38cfa6e06f2eaa3f73ef59dbdbb80e88ca120e0ad6e2ba7059de3112af488e` |
| `insts_lfm3_dec_fa_fin_c32.bin` | `309778b358fd58fe066865a23f577e763d4b4509e45f35a3888fbcd274bb8315` |
| `insts_lfm3_dec_fa_fin_c4.bin` | `24084caf9858bda35db0ce29a3287f188acecce18a97a629aa29f933f23776a9` |
| `insts_lfm3_dec_fa_fin_c48.bin` | `8c00892afc1015bd496ca16d284285d29c60f4f5af3b18ece80dd7306d9328f3` |
| `insts_lfm3_dec_fa_fin_c64.bin` | `31d5096a573b72491b28853bc597fd7def7b73f50345d756b22eece061e821d0` |
| `insts_lfm3_dec_fa_fin_c8.bin` | `99aa6596e8d76e1c9803d55787a9ee18a0703759762b8989fc0edf33f29cad41` |
| `insts_lfm3_dec_fa_fin_c96.bin` | `419e0ec2cb454d53bd25e487f9704fd27301d01ab4ed687220a13595ac220971` |
| `insts_lfm3_dec_fa_part_c256.bin` | `54cb6c416be9bd72e82a8d85be00d3624bea69ce2c39ba879e7dd70b605e079e` |
| `insts_lfm3_dec_phase_a.bin` | `92e4f34ddb884cb588ce61001b2d9d8a411cb3fd61ed7528447d0a7dc9211608` |
| `q4layer_lfm3.xclbin` | `28a518d5c6c4b513d076031603b1c781c052345a40b21efe3e0c27003babb43f` |
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
| `insts_lfm2_dec_attn_c0.bin` | `c5ec41bc5002f3c5dec0b70a27dced6a61666026a6327bb09b6415114ef73a46` |
| `insts_lfm2_dec_attn_c1.bin` | `1dcbbf71c9ecc21286a562a16fd7e4dee6d5dc201dd478618625060bf0235d89` |
| `insts_lfm2_dec_attn_c12.bin` | `f8ffc0151be16ab2d42c1c7c9694ef6ebd317d2f1f16e4466574414774d50d49` |
| `insts_lfm2_dec_attn_c128.bin` | `dc1adaa1c5ee98655c495726e81012b0174a7b860d64c5d9cf8f61336223c1c2` |
| `insts_lfm2_dec_attn_c16.bin` | `49c278e050a52ee42ad1d6d68db057bcad66f2292f222d1cde18bc212b2c567f` |
| `insts_lfm2_dec_attn_c192.bin` | `594a1f6f20dc09586063b69859ea06c513912bb16b571e9d3edec2a128d6f00b` |
| `insts_lfm2_dec_attn_c2.bin` | `6f20c2a19b0c0d8b8cd709a8a31b17e39009cadf0bf78d1129431a7c59c8f0e9` |
| `insts_lfm2_dec_attn_c24.bin` | `2cc196bcc20dd3d5f0cb82b8ad207dc17bc6ec92046a1dab7f2fb0e3fef2c092` |
| `insts_lfm2_dec_attn_c255.bin` | `5ea6fe09bdcec9712ad54acd45df45ba837113f3fd205a327425e2e5ae370c25` |
| `insts_lfm2_dec_attn_c32.bin` | `e72b5686be219ef362019f4ba870ed0b921d0b8d679b2d70b7ded44dadbca038` |
| `insts_lfm2_dec_attn_c4.bin` | `4dccf2ccffaf496ef18c1b7ad7a7668d34f15db43584a2098de3827c123f4ec0` |
| `insts_lfm2_dec_attn_c48.bin` | `3a7556a6881eae953d0afc29342d2d82abbf3eb189b8bba816e7088688040933` |
| `insts_lfm2_dec_attn_c64.bin` | `a7de2b639af2b58a3bc32f844ea4d49d70fd35dcfac4113678f132eb59e18cd9` |
| `insts_lfm2_dec_attn_c8.bin` | `f9a92d450f12032cbbad4a6bb8200624fed61cc5140110c9cba180943330cf61` |
| `insts_lfm2_dec_attn_c96.bin` | `bb53deaa90992a8d60c5bc08e99c367e1b77c5c1ae0a306dca80ed1a00d287bd` |
| `insts_lfm2_dec_conv.bin` | `0d9897f5414106a972cc44def7f5b85fc900d6f7dd18d5801a4d1387cf996ed1` |
| `insts_lfm2_dec_fa_fin_c0.bin` | `49f84b0deef58c9d0651e550f242b985b8176f14a3761f4858ca2c05bcd05a5c` |
| `insts_lfm2_dec_fa_fin_c1.bin` | `a88a6dd5eabce9e061f4f608ed4f89b3e630ccd236d5d41db595cf6e13dffee4` |
| `insts_lfm2_dec_fa_fin_c12.bin` | `89264bf2ebdaa0262000337eef9b427a35532acf83c328630cb2eee7cd24ae0d` |
| `insts_lfm2_dec_fa_fin_c128.bin` | `98d3432ca566f3ada03b37082d8f9b11611c310b4d583e6ec7055810646347b3` |
| `insts_lfm2_dec_fa_fin_c16.bin` | `91633d00d84cae3f8f9c4cd66dea02176e5868379e15aa8335595231880cbc21` |
| `insts_lfm2_dec_fa_fin_c192.bin` | `7d918eab9d6371f5b7a1bdfc59c88e5972a862278e125565b9966b15c6b57151` |
| `insts_lfm2_dec_fa_fin_c2.bin` | `d4663ceebb11a4693589500588847b14d53275f1af7093a2a0f235571af423fb` |
| `insts_lfm2_dec_fa_fin_c24.bin` | `9d801f1961d6c0fc08a12e4d5ac6813ecf096723fe59a926f964c126e4d69238` |
| `insts_lfm2_dec_fa_fin_c255.bin` | `662ad0b8422180e0e0a4a26bb7662dcf060795bdd232b279b2325295237e9eb5` |
| `insts_lfm2_dec_fa_fin_c32.bin` | `1502dc54d2f0fcfff3f8b7ced30d9050655606dd4c3cedcf8af67f73ed8e9714` |
| `insts_lfm2_dec_fa_fin_c4.bin` | `e62550b13fa833f621b51746bbf4ffdd90730491e77468cb91c4903ddf0530bd` |
| `insts_lfm2_dec_fa_fin_c48.bin` | `881123020d870be1b9fd6e46a59d6b110c4d912cd73372742f6a893c9d586037` |
| `insts_lfm2_dec_fa_fin_c64.bin` | `92302a050bf699271f25c4ee5d6d37fd9619b66898595c015a5f0fa0debf784f` |
| `insts_lfm2_dec_fa_fin_c8.bin` | `4cb4b1ed96fd135682062906c18c2a4152773cfa819d3f699c884f0cff1c7651` |
| `insts_lfm2_dec_fa_fin_c96.bin` | `f7e4e599c25ded55beafd9a0381813af3224a577d4639f8c3aa92dcac338132e` |
| `insts_lfm2_dec_fa_part_c256.bin` | `7b3f7c13ec039cef3617faae123207db5fea5da4741b45a6e233a60ff72691c1` |
| `insts_lfm2_dec_phase_a.bin` | `79b86f29bb8fd0a679f47db547a78ec52edfa38e509ae352007b493ca2bf85b3` |
| `q4layer_lfm2.xclbin` | `698ea3ae4cef2b69e6213d8013dfaff88027bb88294ac85640c4bf60768cc426` |
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
