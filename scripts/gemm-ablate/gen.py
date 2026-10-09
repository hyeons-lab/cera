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

# ---- register-blocked rewrite: R weight rows per thread, 256/R threads per workgroup ----
import sys
def gen(R, name, dequant=True):
    T = 256 // R
    L = []
    L.append('''struct StreamParams {
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
''')
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
    L.append(f'        for (uint s = 0u; s < {2 * R}u; s++) {{')
    L.append(f'            uint e = lid * {2 * R}u + s;')
    L.append('            uint kk = e / 8u;')
    L.append('            uint g = e % 8u;')
    L.append('            uint b = (kb * 64u + kk) * P.n_pad + col_base + g * 4u;')
    L.append('            sh_b[kk * 8u + g] = half4(src_b[b], src_b[b + 1u], src_b[b + 2u], src_b[b + 3u]);')
    L.append('        }')
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
