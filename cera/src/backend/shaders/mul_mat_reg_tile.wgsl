// Register-tiled matmul kernel: dst = src0 * src1
//
// src0 (weights): [m, k] row-major, dense or quantized
// src1 (activations): n column-vectors with params.x_stride floats each
// dst (output): n column-vectors with params.y_stride floats each
//
// Tiling strategy:
// Each workgroup covers TILE_ROWS = WORKGROUP_SIZE_M * TILE_M rows of dst and
// TILE_COLS = WORKGROUP_SIZE_N * TILE_N cols. Each thread owns a TILE_M x TILE_N
// register accumulator, held as TILE_N `vec4<f32>`s over the row axis.
//
// TILE_M and TILE_N are fixed at 4 and the inner loop is hand-unrolled for that
// shape. They remain #defines because the host derives its dispatch grid from
// the same constants. A different value does NOT fail to compile — it silently
// computes a fraction of the tile the host dispatched for, which is why both
// call sites carry a `const _: () = assert!(..)` on them (gpu_lfm2.rs,
// vision_encoder_gpu.rs). See the accumulator note below for why this is
// deliberately not parameterized.
//
// PERFORMANCE NOTE — accumulators must be named registers, not an array.
// This kernel previously held `acc` as `array<array<f32, TILE_N>, TILE_M>`
// indexed by the unrolled loop variables. naga emits such a local in the
// `thread` address space, and when the backend compiler declines to fully
// promote it, every accumulator update becomes a device-memory round-trip: the
// kernel measured 76 GFLOP/s on an M1 Max (~0.7% of f32 peak, ~290 cycles per
// FMA), and *shrinking* the register tile made it faster — the inverse of how a
// register tile is supposed to behave. Naming the four accumulators and
// unrolling by hand took the same shapes to 870-1360 GFLOP/s, bit-for-bit
// identical output. Do not reintroduce a loop-variable-indexed local array in
// the inner loop. `cera/examples/wgpu_gemm_bench.rs` measures this directly.
//
// The src0 tile is staged k-major (`sa[k][m]`) so the TILE_M operand reads for
// one k are consecutive rather than strided. The transposing write during
// staging is the cheaper side of that trade — it happens once per k per row,
// against TILE_COLS reads.
//
// OOB note, two separate mechanisms:
//   * What lands in shmem is bounds-checked. Both loaders compare against
//     params.m / params.n / params.k per element and store 0.0 past the edge, so
//     an overhanging tile contributes nothing and the k loop can run the full
//     TILE_K with no tail check. `store_col` likewise guards its rows/column.
//   * The *loads* that feed those checks are issued unconditionally, so a tile
//     overhanging the buffer relies on WebGPU robust buffer access to return
//     zeros rather than fault. That is a spec guarantee; trust it on the target
//     adapter. The full list of reads that resolve before their guard does:
//     the FLOAT loader's `select`, `init_shmem_src1`'s `select`, and in Q4_0 all
//     three of `load_src0_f32_at(base)` (the f16 scale, inside a `select`) plus
//     the two `load_src0_u32_at` quant-word reads.

#define BYTE_HELPERS
#include "common_decls.tmpl"

struct MulMatParams {
    m: u32,
    k: u32,
    n: u32,
    x_stride: u32,
    y_stride: u32,
};

@group(0) @binding(0) var<storage, read> src0: array<SRC0_INNER_TYPE>;
@group(0) @binding(1) var<storage, read> src1: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> params: MulMatParams;

const TOTAL_WORKGROUP_SIZE: u32 = WORKGROUP_SIZE_M * WORKGROUP_SIZE_N;
/// Rows / columns of dst covered by one workgroup.
const TILE_ROWS: u32 = WORKGROUP_SIZE_M * TILE_M;
const TILE_COLS: u32 = WORKGROUP_SIZE_N * TILE_N;
// Pad each staged k-row by 4 floats: keeps every row 4-aligned for the operand
// reads while shifting the bank index between consecutive k, which is the axis
// the transposing store during staging walks across.
const SA_STRIDE: u32 = TILE_ROWS + 4u;
const SB_STRIDE: u32 = TILE_COLS + 4u;

var<workgroup> sa: array<f32, TILE_K * SA_STRIDE>;
var<workgroup> sb: array<f32, TILE_K * SB_STRIDE>;

// Provides `init_shmem_src0` for the selected src0 dtype; it writes through
// `store_sa` below.
#include "mul_mat_decls.tmpl"

/// Stage the src1 (activation) tile, k-major.
fn init_shmem_src1(thread_id: u32, offset_n: u32, k_outer: u32) {
    for (var i = thread_id; i < TILE_COLS * TILE_K; i += TOTAL_WORKGROUP_SIZE) {
        let tile_n = i / TILE_K;
        let tile_k = i % TILE_K;
        let global_n = offset_n + tile_n;
        let global_k = k_outer + tile_k;
        sb[tile_k * SB_STRIDE + tile_n] = select(
            0.0,
            src1[global_n * params.x_stride + global_k],
            global_n < params.n && global_k < params.k,
        );
    }
}

/// Write one column of the register tile: TILE_M consecutive rows of `col`.
/// dst is column-major with stride y_stride, so those rows are adjacent.
fn store_col(col: u32, row: u32, v: vec4<f32>) {
    if (col >= params.n) {
        return;
    }
    let base = col * params.y_stride + row;
    if (row + 3u < params.m) {
        dst[base] = v.x;
        dst[base + 1u] = v.y;
        dst[base + 2u] = v.z;
        dst[base + 3u] = v.w;
    } else {
        if (row < params.m) { dst[base] = v.x; }
        if (row + 1u < params.m) { dst[base + 1u] = v.y; }
        if (row + 2u < params.m) { dst[base + 2u] = v.z; }
        if (row + 3u < params.m) { dst[base + 3u] = v.w; }
    }
}

@compute @workgroup_size(TOTAL_WORKGROUP_SIZE)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let thread_id = local_id.x;
    let local_m = thread_id % WORKGROUP_SIZE_M;
    let local_n = thread_id / WORKGROUP_SIZE_M;

    let wg_m_count = (params.m + TILE_ROWS - 1u) / TILE_ROWS;

    // Linearize wg_id so callers can dispatch as 2D (avoids the 65535
    // num_wg.x limit) without changing the kernel.
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let offset_m = (wg_linear % wg_m_count) * TILE_ROWS;
    let offset_n = (wg_linear / wg_m_count) * TILE_COLS;

    // accN holds TILE_M consecutive ROWS of output column
    // (offset_n + local_n*TILE_N + N) — the layout dst wants.
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    let m0 = local_m * TILE_M;
    let n0 = local_n * TILE_N;

    for (var k_outer = 0u; k_outer < params.k; k_outer += TILE_K) {
        init_shmem_src0(thread_id, offset_m, k_outer);
        init_shmem_src1(thread_id, offset_n, k_outer);

        workgroupBarrier();

        for (var k_inner = 0u; k_inner < TILE_K; k_inner++) {
            let ai = k_inner * SA_STRIDE + m0;
            let bi = k_inner * SB_STRIDE + n0;
            let a = vec4<f32>(sa[ai], sa[ai + 1u], sa[ai + 2u], sa[ai + 3u]);
            acc0 += a * sb[bi];
            acc1 += a * sb[bi + 1u];
            acc2 += a * sb[bi + 2u];
            acc3 += a * sb[bi + 3u];
        }

        workgroupBarrier();
    }

    let row = offset_m + m0;
    let col = offset_n + n0;
    store_col(col, row, acc0);
    store_col(col + 1u, row, acc1);
    store_col(col + 2u, row, acc2);
    store_col(col + 3u, row, acc3);
}
