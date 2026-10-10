#!/usr/bin/env python3
"""Ablation and register-blocking variants of the production prefill GEMM (`gemm_stream_q4_0_k64.slang`).

Writes the variants next to each other in OUT, compiles them with slangc 2026.19 when it is on PATH (or
SLANGC), and prints the A/B command for the phone:

    python3 scripts/gemm-ablate/gen.py /tmp/ablate
    adb push /tmp/ablate /data/local/tmp/ablate
    adb shell 'cd /data/local/tmp/cmp-cera && ./cera gemm-bench --shapes "4608,512,1024 1024,512,4608" \
        --spv /data/local/tmp/ablate/a0_base.spv --spv /data/local/tmp/ablate/a3_nolds_nodequant.spv ...'

Every variant keeps the bindings, grid and parameter block of the production kernel, so `gemm-bench --spv`
times them against it. The ablations a1..a7 compute wrong values on purpose (they remove a component to
time what it costs); a0, r1, r2, r4 and e1..e3 are bit-exact against the production kernel (`vs_base=0`).
Results are in benchmarks/ANDROID_VL_IMAGE.md ("What holds the GEMM at 31%").
"""
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = open(os.path.join(ROOT, "cera/src/backend/shaders/spirv/gemm_stream_q4_0_k64.slang")).read()
OUT = sys.argv[1] if len(sys.argv) > 1 else "ablate"
os.makedirs(OUT, exist_ok=True)

# ---- staging of the 64 (k) x 32 (columns) B tile into shared memory ----
def stage_lines(threads, xf32):
    """The loop staging the tile into `sh_b[kk * 8 + g]` (4 columns g*4..g*4+3 of row kk, as half4).

    Default: from the transposed f16 copy `src_b` ([k][n_pad], written by `transpose_cast_f16`).
    `xf32`: straight from the f32 token-major activations `src_bf` ([n][k]) the previous pass wrote,
    converting as it goes, so no transpose pass is needed: lane-consecutive threads read consecutive
    k of one token (coalesced), and a row past the last token is clamped onto it (its columns are
    never stored).
    """
    stage = 512 // threads
    if not xf32:
        return [
            f"        for (uint s = 0u; s < {stage}u; s++) {{",
            f"            uint e = lid * {stage}u + s;",
            "            uint kk = e / 8u;",
            "            uint g = e % 8u;",
            "            uint b = (kb * 64u + kk) * P.n_pad + col_base + g * 4u;",
            "            sh_b[kk * 8u + g] = half4(src_b[b], src_b[b + 1u], src_b[b + 2u], src_b[b + 3u]);",
            "        }",
        ]
    return [
        f"        for (uint s = 0u; s < {stage}u; s++) {{",
        f"            uint e = s * {threads}u + lid;",
        "            uint kk = e % 64u;",
        "            uint g = e / 64u;",
        "            uint t0 = col_base + g * 4u;",
        "            uint kidx = kb * 64u + kk;",
        "            uint last = P.n_valid - 1u;",
        "            sh_b[kk * 8u + g] = half4(half(src_bf[min(t0, last) * P.k + kidx]),",
        "                                      half(src_bf[min(t0 + 1u, last) * P.k + kidx]),",
        "                                      half(src_bf[min(t0 + 2u, last) * P.k + kidx]),",
        "                                      half(src_bf[min(t0 + 3u, last) * P.k + kidx]));",
        "        }",
    ]


# ---- register-blocked rewrite: R weight rows per thread, 256/R threads per workgroup ----
import sys
def gen(R, name, dequant=True, header='', xf32=False):
    T = 256 // R
    L = [header] if header else []
    decls = '''struct StreamParams {
    uint m;
    uint k;
    uint n_valid;
    uint n_pad;
    uint y_stride;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint> src_q;
[[vk::binding(1, 0)]] StructuredBuffer<uint> src_d;
[[vk::binding(2, 0)]] StructuredBuffer<half> src_b;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> dst;
[[vk::binding(4, 0)]] StructuredBuffer<StreamParams> paramsBuf;

groupshared half4 sh_b[64 * 8];
'''
    L.append(decls.replace("StructuredBuffer<half> src_b;", "StructuredBuffer<float> src_bf;") if xf32 else decls)
    L.append(f'[numthreads({T}, 1, 1)]')
    L.append('void main(uint3 gid : SV_GroupID, uint lid : SV_GroupThreadID) {')
    L.append('    let P = paramsBuf[0];')
    L.append('    uint col_base = gid.y * 32u;')
    for r in range(R):
        L.append(f'    uint row{r} = gid.x * 256u + {r * T}u + lid;')
        L.append(f'    uint rrow{r} = row{r} < P.m ? row{r} : 0u;')
        for i in range(8):
            L.append(f'    half4 g{r}_{i} = half4(0.0h);')
    L.append('    uint n_kb = P.k / 64u;')
    L.append('    for (uint kb = 0u; kb < n_kb; kb++) {')
    L.extend(stage_lines(T, xf32))
    L.append('        GroupMemoryBarrierWithGroupSync();')
    for r in range(R):
        L.append(f'        uint pd{r} = src_d[kb * P.m + rrow{r}];')
        L.append(f'        half dw{r}_0 = half(f16tof32(pd{r} & 0xFFFFu));')
        L.append(f'        half dw{r}_1 = half(f16tof32(pd{r} >> 16u));')
    L.append('        for (uint p = 0u; p < 8u; p++) {')
    for r in range(R):
        L.append(f'            uint pq{r} = src_q[(kb * 8u + p) * P.m + rrow{r}];')
        L.append(f'            uint w4_lo{r} = pq{r} & 0xFFFFu;')
        L.append(f'            uint w4_hi{r} = pq{r} >> 16u;')
        L.append(f'            half dw{r} = (p < 4u ? dw{r}_0 : dw{r}_1);')
    L.append('            uint kl0 = p * 8u;')
    for t in range(8):
        part = 'lo' if t < 4 else 'hi'
        sh = (t % 4) * 4
        L.append('            {')
        L.append(f'                uint b = kl0 * 8u + {t * 8}u;')
        for i in range(8):
            L.append(f'                half4 b{i} = sh_b[b + {i}u];')
        for r in range(R):
            nib = f'(w4_{part}{r} >> {sh}u) & 0xFu' if sh else f'w4_{part}{r} & 0xFu'
            if dequant:
                L.append(f'                half w{r} = (half({nib}) - half(8.0h)) * dw{r};')
            else:
                L.append(f'                half w{r} = dw{r};')
            for i in range(8):
                L.append(f'                g{r}_{i} += b{i} * w{r};')
        L.append('            }')
    L.append('        }')
    L.append('        GroupMemoryBarrierWithGroupSync();')
    L.append('    }')
    for r in range(R):
        L.append(f'    if (row{r} < P.m) {{')
        for i in range(8):
            L.append(f'        uint c{i} = col_base + {4 * i}u;')
            L.append(f'        if (c{i} < P.n_valid) {{')
            L.append(f'            float4 f = float4(g{r}_{i});')
            L.append(f'            dst[c{i} * P.y_stride + row{r}] = f.x;')
            L.append(f'            if (c{i} + 1u < P.n_valid) {{ dst[(c{i} + 1u) * P.y_stride + row{r}] = f.y; }}')
            L.append(f'            if (c{i} + 2u < P.n_valid) {{ dst[(c{i} + 2u) * P.y_stride + row{r}] = f.z; }}')
            L.append(f'            if (c{i} + 3u < P.n_valid) {{ dst[(c{i} + 3u) * P.y_stride + row{r}] = f.w; }}')
            L.append('        }')
        L.append('    }')
    L.append('}')
    open(name, 'w').write('\n'.join(L) + '\n')


# ---- fused gate/up GEMM: one B tile, two weight rows per thread, SiLU(gate) * up epilogue ----
def gen_gateup(name, header="", threads=128, xf32=False):
    """One thread owns row `gid.x * threads + lid` of both the gate and the up weights."""
    L = [header] if header else []
    decls = """struct StreamParams {
    uint m;
    uint k;
    uint n_valid;
    uint n_pad;
    uint y_stride;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint> gate_q;
[[vk::binding(1, 0)]] StructuredBuffer<uint> gate_d;
[[vk::binding(2, 0)]] StructuredBuffer<uint> up_q;
[[vk::binding(3, 0)]] StructuredBuffer<uint> up_d;
[[vk::binding(4, 0)]] StructuredBuffer<half> src_b;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> dst;
[[vk::binding(6, 0)]] StructuredBuffer<StreamParams> paramsBuf;

groupshared half4 sh_b[64 * 8];
"""
    L.append(decls.replace("StructuredBuffer<half> src_b;", "StructuredBuffer<float> src_bf;") if xf32 else decls)
    L.append(f"[numthreads({threads}, 1, 1)]")
    L.append("void main(uint3 gid : SV_GroupID, uint lid : SV_GroupThreadID) {")
    L.append("    let P = paramsBuf[0];")
    L.append(f"    uint row = gid.x * {threads}u + lid;")
    L.append("    uint col_base = gid.y * 32u;")
    L.append("    uint rrow = row < P.m ? row : 0u;")
    L.append("    bool row_valid = row < P.m;")
    for i in range(8):
        L.append(f"    half4 g{i} = half4(0.0h);")
        L.append(f"    half4 u{i} = half4(0.0h);")
    L.append("    uint n_kb = P.k / 64u;")
    L.append("    for (uint kb = 0u; kb < n_kb; kb++) {")
    L.extend(stage_lines(threads, xf32))
    L.append("""        GroupMemoryBarrierWithGroupSync();
        uint pdg = gate_d[kb * P.m + rrow];
        uint pdu = up_d[kb * P.m + rrow];
        half dwg0 = half(f16tof32(pdg & 0xFFFFu));
        half dwg1 = half(f16tof32(pdg >> 16u));
        half dwu0 = half(f16tof32(pdu & 0xFFFFu));
        half dwu1 = half(f16tof32(pdu >> 16u));
        for (uint p = 0u; p < 8u; p++) {
            uint pqg = gate_q[(kb * 8u + p) * P.m + rrow];
            uint pqu = up_q[(kb * 8u + p) * P.m + rrow];
            uint glo = pqg & 0xFFFFu;
            uint ghi = pqg >> 16u;
            uint ulo = pqu & 0xFFFFu;
            uint uhi = pqu >> 16u;
            half dwg = (p < 4u ? dwg0 : dwg1);
            half dwu = (p < 4u ? dwu0 : dwu1);
            uint kl0 = p * 8u;""")
    for t in range(8):
        part = "lo" if t < 4 else "hi"
        sh = (t % 4) * 4
        L.append("            {")
        L.append(f"                uint b = kl0 * 8u + {t * 8}u;")
        for i in range(8):
            L.append(f"                half4 b{i} = sh_b[b + {i}u];")
        for who, w in (("g", "dwg"), ("u", "dwu")):
            nib = f"({who}{part} >> {sh}u) & 0xFu" if sh else f"{who}{part} & 0xFu"
            L.append(f"                half w{who} = (half({nib}) - half(8.0h)) * {w};")
            for i in range(8):
                L.append(f"                {who}{i} += b{i} * w{who};")
        L.append("            }")
    L.append("        }")
    L.append("        GroupMemoryBarrierWithGroupSync();")
    L.append("    }")
    L.append("    if (!row_valid) {")
    L.append("        return;")
    L.append("    }")
    for i in range(8):
        L.append(f"    uint c{i} = col_base + {4 * i}u;")
        L.append(f"    if (c{i} < P.n_valid) {{")
        L.append(f"        float4 cg = clamp(float4(g{i}), float4(-80.0), float4(80.0));")
        L.append(f"        float4 o = (cg / (float4(1.0) + exp(-cg))) * float4(u{i});")
        L.append(f"        dst[c{i} * P.y_stride + row] = o.x;")
        L.append(f"        if (c{i} + 1u < P.n_valid) {{ dst[(c{i} + 1u) * P.y_stride + row] = o.y; }}")
        L.append(f"        if (c{i} + 2u < P.n_valid) {{ dst[(c{i} + 2u) * P.y_stride + row] = o.z; }}")
        L.append(f"        if (c{i} + 3u < P.n_valid) {{ dst[(c{i} + 3u) * P.y_stride + row] = o.w; }}")
        L.append("    }")
    L.append("}")
    open(name, "w").write("\n".join(L) + "\n")


# ---- ablations of the production source: each removes one component ----
def nodq(s):
    return s.replace("half w = (half(nib) - half(8.0h)) * dw;", "half w = dw;")


def nolds(s):
    s = re.sub(r"sh_b\[b \+ (\d)u\]", r"bv\1", s)
    decl = "".join(f"        half4 bv{i} = half4(half(kb + {i}u));\n" for i in range(8))
    a = "        half dw1 = half(f16tof32(pd0 >> 16u));\n"
    assert a in s
    return s.replace(a, a + decl)


def nostage(s):
    s2 = re.sub(r"        // Stage the B tile.*?GroupMemoryBarrierWithGroupSync\(\);\n", "", s, flags=re.S)
    assert s2 != s
    return s2


def noglobal(s):
    a = "uint pq = src_q[(kb * 8u + p) * P.m + rrow];"
    b = "uint pd0 = src_d[(kb * 1u + 0u) * P.m + rrow];"
    assert a in s and b in s
    return s.replace(a, "uint pq = (kb * 8u + p) * 0x9E3779B1u + rrow;").replace(b, "uint pd0 = kb * 0x3C003C00u + rrow;")


def magic(s):
    a = "half w = (half(nib) - half(8.0h)) * dw;"
    assert a in s
    return s.replace(a, "half w = (reinterpret<half>(uint16_t(nib | 0x6400u)) - half(1032.0h)) * dw;")


def stage64(s):
    s = s.replace("[[vk::binding(2, 0)]] StructuredBuffer<half> src_b;", "[[vk::binding(2, 0)]] StructuredBuffer<uint2> src_b2;")
    b = "sh_b[kk * 8u + g] =\n                half4(src_b[b], src_b[b + 1u], src_b[b + 2u], src_b[b + 3u]);"
    assert b in s
    return s.replace(b, "sh_b[kk * 8u + g] = reinterpret<half4>(src_b2[b / 4u]);")


VARIANTS = {
    "a0_base": lambda s: s,
    "a1_nodequant": nodq,
    "a2_nolds": nolds,
    "a3_nolds_nodequant": lambda s: nolds(nodq(s)),
    "a4_nostage": nostage,
    "a5_noglobal": noglobal,
    "a6_fmaonly": lambda s: nostage(noglobal(nolds(nodq(s)))),
    "a7_ldsfma": lambda s: noglobal(nodq(s)),
    "e1_magic": magic,
    "e2_stage64": stage64,
}
for name, f in VARIANTS.items():
    open(os.path.join(OUT, name + ".slang"), "w").write(f(SRC))
os.chdir(OUT)
gen(1, "r1.slang")
gen(2, "r2.slang")
gen(4, "r4.slang")
gen(2, "r2_nodq.slang", False)

slangc = os.environ.get("SLANGC") or shutil.which("slangc") or os.path.expanduser("~/.local/slang/bin/slangc")
for f in sorted(os.listdir(".")):
    if f.endswith(".slang"):
        r = subprocess.run([slangc, f, "-target", "spirv", "-O3", "-entry", "main", "-stage", "compute", "-o", f[:-6] + ".spv"],
                           capture_output=True, text=True)
        print(("ok   " if r.returncode == 0 else "FAIL ") + f)


# ---- production kernels: `gen.py --production` rewrites them next to the kernel they extend ----
# All three read the f32 token-major activations directly (`xf32`), so the host runs no
# `transpose_cast_f16` pass for them. GATEUP_THREADS is the fused kernel's workgroup size (rows per
# workgroup); the host dispatches ceil(m / GATEUP_THREADS) groups in X (`GATEUP_ROWS_PER_GROUP` in
# gpu_lfm2.rs).
GATEUP_THREADS = int(os.environ.get("GATEUP_THREADS", "128"))
if "--production" in sys.argv:
    prod = os.path.join(ROOT, "cera/src/backend/shaders/spirv")
    common = """// The activations come straight from the previous pass: binding 2 is the f32 token-major
// `x[n][k]` (x_stride == k), converted to f16 while the B tile is staged, so there is no transpose
// pass and no f16 copy. Rows past the last token are clamped onto it (their columns are never
// stored). Staging this way is also 8 to 20% faster than reading a transposed f16 copy on an
// Adreno 830.
"""
    hdr_r1 = """// Streaming Q4_0 prefill GEMM, k-slice-64, one weight row per thread, B staged from f32 token-major
// activations. GENERATED by scripts/gemm-ablate/gen.py --production; edit the generator, not this file.
//
// Same weights, parameter block, entry (`main`) and grid (ceil(m/256), n_pad/32) as
// `gemm_stream_q4_0_k64.slang`, which it replaces on the direct-B path, and bit-exact results.
""" + common
    hdr_r2 = """// Streaming Q4_0 prefill GEMM, k-slice-64, two weight rows per thread, B staged from f32 token-major
// activations. GENERATED by scripts/gemm-ablate/gen.py --production; edit the generator, not this file.
//
// Same bindings, parameter block, entry (`main`) and grid (ceil(m/256), n_pad/32) as
// `gemm_stream_q4_0_k64_xf32.slang`, and bit-exact results, but a workgroup is 128 threads and each
// thread owns two rows (row = group*256 + lid and +128) over the same 32 columns. The B tile is read
// from shared memory once per k and feeds two rows, which halves the shared-memory reads per FMA:
// those were 43% of the one-row kernel's time on an Adreno 830. Measured 5 to 20% faster at 64 or
// more workgroups (m/256 * n_pad/32); below that the smaller workgroups under-fill the GPU and the
// one-row kernel wins, so the host picks by workgroup count (`gemm_stream_two_row`).
""" + common
    hdr_gu = f"""// Fused gate/up streaming Q4_0 prefill GEMM with a SiLU epilogue, k-slice-64, B staged from f32
// token-major activations. GENERATED by scripts/gemm-ablate/gen.py --production; edit the generator,
// not this file.
//
// One thread owns the same row of the gate and the up weights over 32 columns: both read each B
// tile from shared memory once, and the epilogue stores silu(gate) * up (gate clamped to +-80, as
// `silu_mul_inplace` does) instead of two projections and a separate pass. Workgroups of
// {GATEUP_THREADS} threads cover {GATEUP_THREADS} rows: grid (ceil(m/{GATEUP_THREADS}), n_pad/32). The output has
// the layout of the gate GEMM's (dst[col * y_stride + row]). Bindings: 0/1 gate q/d, 2/3 up q/d,
// 4 f32 token-major activations, 5 dst, 6 params.
""" + common
    os.chdir(prod)
    gen(1, "gemm_stream_q4_0_k64_xf32.slang", True, hdr_r1, xf32=True)
    gen(2, "gemm_stream_q4_0_k64_r2.slang", True, hdr_r2, xf32=True)
    gen_gateup("gemm_stream_q4_0_k64_gateup.slang", hdr_gu, GATEUP_THREADS, xf32=True)
    slangc = os.environ.get("SLANGC") or shutil.which("slangc") or os.path.expanduser("~/.local/slang/bin/slangc")
    for f in ("gemm_stream_q4_0_k64_xf32", "gemm_stream_q4_0_k64_r2", "gemm_stream_q4_0_k64_gateup"):
        r = subprocess.run([slangc, f + ".slang", "-target", "spirv", "-O3", "-entry", "main", "-stage", "compute", "-o", f + ".spv"],
                           capture_output=True, text=True)
        print(("ok   " if r.returncode == 0 else "FAIL ") + f, r.stderr[:300])
