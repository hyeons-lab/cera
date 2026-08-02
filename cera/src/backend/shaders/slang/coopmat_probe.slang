// Probe: can one Slang source carry a Metal `simdgroup_matrix` path *and* a
// portable path, given WGSL has no cooperative-matrix type at all?
//
// This is not a production kernel and nothing dispatches it. It exists because
// the answer decides whether the eight `simdgroup_matrix` GEMMs (the hot path,
// and the bulk of what a Slang migration would have to absorb) are reachable at
// all, and because the answer is not obvious from any documentation: it depends
// on whether `__target_switch` eliminates a branch *before* entry-point
// capability validation runs, or after.
//
// It does eliminate first. Compiling this to `metal` yields real hardware MMA
// (`simdgroup_load` / `simdgroup_multiply_accumulate` / `simdgroup_store`);
// compiling the same source to `wgsl` yields the scalar fallback with no trace
// of the cooperative-matrix path, rather than the
// `E36107 unavailable features in entry point` that the unguarded version
// raises. `tests/slang_multitarget_parity.rs` pins both halves of that.
//
// ## Why this is a probe and not a GEMM
//
// A fixed 8x8 with no tiling, no threadgroup staging, no ragged edges and no
// dequant. A real port of `gemm_q*.metal` shares far less between the two
// branches than `softmax.slang` does: the Metal side would be simdgroup tiling
// and the portable side the existing reg-tile kernel with its loaders, so the
// win there is the binding contract and the dequant math, not the kernel body.
// Establishing the capability is the cheap part; that port is not.
//
// ## The f16 hazard this exposed
//
// Typing the operands `half` makes the WGSL emission open with `enable f16;`.
// cera requests `wgpu::Features::SHADER_F16` only when the adapter reports it
// (`GpuContext::new`), so a generated f16 kernel fails pipeline creation on any
// adapter without it. Nothing dispatches this probe, so it cannot break a
// device; a real port has to either keep f32 operands or gate on the feature.
// This is the first shader in the tree to emit `enable f16` at all.

using namespace linalg;

[[vk::binding(0)]] StructuredBuffer<half>    a_buf : register(t0);
[[vk::binding(1)]] StructuredBuffer<half>    b_buf : register(t1);
[[vk::binding(2)]] RWStructuredBuffer<float> c_buf : register(u2);

[shader("compute")]
[numthreads(32, 1, 1)]
void coopmat_probe(uint3 tid : SV_GroupThreadID) {
    __target_switch {
    case metal:
    {
        // 8x8 is Metal's native simdgroup_matrix tile.
        typealias MatA = CoopMat<half, MemoryScope.Subgroup, 8, 8, CoopMatMatrixUse.MatrixA>;
        typealias MatB = CoopMat<half, MemoryScope.Subgroup, 8, 8, CoopMatMatrixUse.MatrixB>;
        typealias MatC = CoopMat<float, MemoryScope.Subgroup, 8, 8, CoopMatMatrixUse.MatrixAccumulator>;

        MatA a = MatA.Load<CoopMatMatrixLayout.RowMajor>(a_buf, 0, 8);
        MatB b = MatB.Load<CoopMatMatrixLayout.RowMajor>(b_buf, 0, 8);
        MatC c = MatC(0.0f);
        c = coopMatMulAdd<float, false>(a, b, c);
        c.Store<CoopMatMatrixLayout.RowMajor>(c_buf, 0, 8);
        break;
    }
    default:
    {
        // One thread per output element. Correctness here is incidental; the
        // point is that this branch is what survives on a target that cannot
        // express the one above.
        uint t = tid.x;
        if (t < 64u) {
            uint row = t / 8u;
            uint col = t % 8u;
            float acc = 0.0f;
            for (uint i = 0u; i < 8u; i++) {
                acc += float(a_buf[row * 8u + i]) * float(b_buf[i * 8u + col]);
            }
            c_buf[t] = acc;
        }
        break;
    }
    }
}
