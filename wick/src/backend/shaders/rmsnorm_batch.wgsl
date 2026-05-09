// Batched RMSnorm: process N independent vectors in a single dispatch.
// Each workgroup handles one vector (same algorithm as rmsnorm.wgsl).
// Dispatch: (N, 1, 1) workgroups of 256 threads.
//
// Two entry points:
//
//   rmsnorm_batch          — read src, write dst.
//                            dst[i] = src[i] * inv_rms(src) * w[i]
//
//   add_rmsnorm_batch      — read src + residual, write back to src and dst.
//                            src[i] += residual[i];
//                            dst[i] = src[i] * inv_rms(src) * w[i]
//
// Both share the same Params struct and binding layout for slots 0–3;
// `add_rmsnorm_batch` reads its residual from binding 4.
//
// Bind groups:
//   @binding(0) src: array<f32>      (read-write — `add_rmsnorm_batch` writes
//                                      back the post-add value; plain
//                                      `rmsnorm_batch` only reads)
//   @binding(1) dst: array<f32>      (read-write — normalized output)
//   @binding(2) w: array<f32>        (read — per-element scale)
//   @binding(3) params: vec4<u32>    (n, eps_bits, src_stride, dst_stride)
//   @binding(4) residual: array<f32> (read — only used by `add_rmsnorm_batch`,
//                                      stride = src_stride)

@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> params: vec4<u32>;
@group(0) @binding(4) var<storage, read> residual: array<f32>;

var<workgroup> shared_sum: array<f32, 256>;

// Tree-reduce `shared_sum[0..256]` in-place; result lands in `shared_sum[0]`.
// Caller must have already populated `shared_sum[tid]` and issued a
// `workgroupBarrier()` so all writes are visible.
fn workgroup_sum_reduce(tid: u32) {
    if tid < 128u { shared_sum[tid] += shared_sum[tid + 128u]; }
    workgroupBarrier();
    if tid < 64u { shared_sum[tid] += shared_sum[tid + 64u]; }
    workgroupBarrier();
    if tid < 32u { shared_sum[tid] += shared_sum[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { shared_sum[tid] += shared_sum[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u { shared_sum[tid] += shared_sum[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u { shared_sum[tid] += shared_sum[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u { shared_sum[tid] += shared_sum[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u { shared_sum[tid] += shared_sum[tid + 1u]; }
    workgroupBarrier();
}

@compute @workgroup_size(256, 1, 1)
fn rmsnorm_batch(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let n = params.x;
    let eps = bitcast<f32>(params.y);
    let src_off = wid.x * params.z;
    let dst_off = wid.x * params.w;

    // Phase 1: partial sum of squares.
    var partial: f32 = 0.0;
    var i = tid;
    while i < n {
        let v = src[src_off + i];
        partial += v * v;
        i += 256u;
    }
    shared_sum[tid] = partial;
    workgroupBarrier();

    workgroup_sum_reduce(tid);
    let sum_sq = shared_sum[0];
    let inv_rms = 1.0 / sqrt(sum_sq / f32(n) + eps);

    // Phase 2: write normalized values to dst.
    i = tid;
    while i < n {
        dst[dst_off + i] = src[src_off + i] * inv_rms * w[i];
        i += 256u;
    }
}

@compute @workgroup_size(256, 1, 1)
fn add_rmsnorm_batch(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let n = params.x;
    let eps = bitcast<f32>(params.y);
    let src_off = wid.x * params.z;
    let dst_off = wid.x * params.w;
    let res_off = src_off; // residual shares stride with src

    // Phase 1: add residual in-place AND compute sum of squares
    // of the post-add value. Mirrors the metal kernel.
    var partial: f32 = 0.0;
    var i = tid;
    while i < n {
        let v = src[src_off + i] + residual[res_off + i];
        src[src_off + i] = v;
        partial += v * v;
        i += 256u;
    }
    shared_sum[tid] = partial;
    workgroupBarrier();

    workgroup_sum_reduce(tid);
    let sum_sq = shared_sum[0];
    let inv_rms = 1.0 / sqrt(sum_sq / f32(n) + eps);

    i = tid;
    while i < n {
        dst[dst_off + i] = src[src_off + i] * inv_rms * w[i];
        i += 256u;
    }
}
