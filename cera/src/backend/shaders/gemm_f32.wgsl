// Batched f32 GEMM (NT form) for the prefill-path LoRA apply.
//
// The wgpu prefill batch buffers are TOKEN-MAJOR (`[n_tokens × dim]`, row-major —
// token `i`'s channels are contiguous at `i*dim`), so the LoRA factors — `A`
// `[rank × k]` and `B` `[d × rank]`, both row-major — are consumed *transposed*
// (NT form): each output element is a dot of an lhs row with an rhs row. This
// mirrors the base projection GEMMs (`out[tok,o] = Σ W[o,i]·X[tok,i]`).
//
//   C[M×N] = Lhs[M×K] · Rhs[N×K]ᵀ   →   c[i*N+j] = Σ_p lhs[i*K+p] · rhs[j*K+p]
//
// One workgroup (32 threads) per output element; the K reduction is a workgroup
// tree-reduce. A flat output index is split across a 2-D dispatch grid via
// `get_wid` (`wid.x + wid.y*MAX_WG`) so M*N can exceed the 65535 per-dimension
// cap. Correct-first-cut kernel — the LoRA GEMMs are tiny (rank ≤ 64), so this
// isn't a hot path. Mirrors `gemm_f32.metal`.

@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
#ifdef IMMEDIATES
struct ImmParams {
    v: vec4<u32>,
}
var<immediate> imm: ImmParams;
#else
@group(0) @binding(3) var<storage, read> params: vec4<u32>;
#endif

fn get_params() -> vec4<u32> {
#ifdef IMMEDIATES
    return imm.v;
#else
    return params;
#endif
}

#include "common_decls.tmpl"

const WG_SIZE: u32 = 32u;

var<workgroup> partials: array<f32, 32>;

// Dot lhs row `i` against rhs row `j` over K, tree-reduced across the 32-thread
// workgroup. All threads leave with the finished sum in `partials[0]`; the
// caller reads it (only thread 0 writes the output). Barriers sit in uniform
// control flow (the while loop carries none), so this is well-defined on
// adapters whose subgroups are narrower than the workgroup.
fn nt_dot(tid: u32, i: u32, j: u32, k: u32) -> f32 {
    let lhs_off = i * k;
    let rhs_off = j * k;
    var partial = 0.0;
    var c = tid;
    while c < k {
        partial += lhs[lhs_off + c] * rhs[rhs_off + c];
        c += WG_SIZE;
    }
    partials[tid] = partial;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            partials[tid] += partials[tid + stride];
        }
        workgroupBarrier();
    }
    return partials[0];
}

// C = Lhs · Rhsᵀ (overwrite). Used for the LoRA down-projection
// (`Tmp[n×rank] = X[n×k]·Aᵀ`).
@compute @workgroup_size(32, 1, 1)
fn gemm_f32_nt(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let p = get_params();
    let m = p.x;
    let n = p.y;
    let k = p.z;
    let idx = get_wid(wid);
    if idx >= m * n { return; }
    let i = idx / n; // lhs row (token)
    let j = idx % n; // rhs row (output channel)
    let v = nt_dot(lid.x, i, j, k);
    if lid.x == 0u { out[idx] = v; }
}

// Accumulate variant: `C += Lhs * Rhs^T`. Same layout/dispatch as `gemm_f32_nt`;
// used for the LoRA up-projection epilogue, where the `alpha/rank` scale is
// pre-folded into the uploaded `B` so there is no separate scale pass.
@compute @workgroup_size(32, 1, 1)
fn gemm_f32_nt_accum(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let p = get_params();
    let m = p.x;
    let n = p.y;
    let k = p.z;
    let idx = get_wid(wid);
    if idx >= m * n { return; }
    let i = idx / n;
    let j = idx % n;
    let v = nt_dot(lid.x, i, j, k);
    if lid.x == 0u { out[idx] += v; }
}
