// Case table for the fused matmuls sharing one activation (MUL_MAT_NX: a fused Q/K/V or gate/up):
// `htp_kparams_solve_mmnx` on the first weight and the activation, then the DSP-kernel check the
// fusion pass applies (`is_supported_mul_mat_nx_kernel` in `htp-fusion.cpp`, mirrored here; gen.py fails
// when the upstream text drifts: the HMX 2-D kernel or the row-parallel tiled HVX one). Line format:
//   <weight0> | <act> | <n_weights> | <threads> <vtcm> <hmx> <mdev> => <supported 0|1> <params> | unsupported
#include "mm_common.h"

static int n_cases = 0;

static void emit(const ggml_tensor & w, const ggml_tensor & x, int n_weights, const MmCfg & c) {
    ggml_hexagon_session sess = mm_sess(c);
    ggml_tensor w0 = w, x0 = x;
    htp_mm_kernel_params kp;
    const bool ok = htp_kparams_solve_mmnx(&sess, &w0, &x0, n_weights, &kp);

    mm_print_t(w0); printf(" | "); mm_print_t(x0); printf(" | %d | ", n_weights); mm_print_cfg(c); printf(" =>");
    if (ok) {
        const bool tiled = ggml_hexagon_tensor_is_tiled(&w0);
        const bool supported = kp.n_hmx ? kp.kernel_type == HTP_MM_KERNEL_HMX_2D
                                        : tiled && kp.kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW;
        printf(" %d", supported ? 1 : 0);
    }
    mm_print_words(ok, kp);
    printf("\n");
    n_cases++;
}

int main() {
    opt_rt_repack = 0;
    const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K };

    // 1. Fused Q/K/V and gate/up shapes at the shipped session.
    for (ggml_type t : types) {
        const bool kq = (t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q6_K);
        for (int64_t k : { 256, 512, 768, 1024, 2048, 4608 }) {
            if (!kq && k == 768) continue;
            for (int64_t n : { 96, 512, 1024, 2048, 4608, 8192 })
            for (int64_t rows : { 1, 2, 4, 5, 8, 16, 33, 130, 512, 1000, 2048 })
            for (int nw : { 2, 3, 4 }) {
                emit(mm_w(t, k, n), mm_f(k, rows), nw, MM_BASE);
            }
        }
    }

    // 2. Session sweep.
    // a static table: braced lists of compound literals are temporaries that die before the loop body
    static const int64_t SESSION_SHAPES[][2] = { { 1024, 1024 }, { 1024, 4608 }, { 2048, 8192 } };
    for (ggml_type t : types) for (const auto & s : SESSION_SHAPES)
    for (int64_t rows : { 1, 5, 33, 130, 512 }) for (uint32_t th : { 1, 4, 6 }) for (size_t vt : MM_VTCMS)
    for (uint32_t hmx : { 0u, 1u }) for (uint32_t mdev : { 1u, 2u }) {
        emit(mm_w(t, s[0], s[1]), mm_f(s[0], rows), 3, { th, vt, hmx, mdev });
    }

    // 3. Other tiled types and plain F16 weights.
    for (ggml_type t : { GGML_TYPE_Q4_1, GGML_TYPE_Q5_K, GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
                         GGML_TYPE_Q1_0, GGML_TYPE_Q2_0, GGML_TYPE_F16 })
    for (bool tiled : { true, false }) for (int64_t rows : { 1, 5, 64 }) for (uint32_t hmx : { 0u, 1u }) {
        emit(mm_w(t, 1024, 1024, 1, tiled), mm_f(1024, rows), 2, { 8, 8u << 20, hmx, 1 });
    }

    // 4. Activation variants: F16, narrower than the weight, padded or transposed rows, batched.
    for (ggml_type t : { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K }) for (int64_t rows : { 1, 5, 33 })
    for (uint32_t hmx : { 0u, 1u }) {
        const MmCfg c = { 8, 8u << 20, hmx, 1 };
        emit(mm_w(t, 1024, 1024), mm_f(1024, rows, 1, 0, GGML_TYPE_F16), 3, c);
        emit(mm_w(t, 1024, 1024), mm_f(1000, rows), 3, c);
        emit(mm_w(t, 1024, 1024), mm_f(1024, rows, 1, 1), 3, c);
        emit(mm_w(t, 1024, 1024), mm_f(1024, rows, 1, 2), 3, c);
        emit(mm_w(t, 1024, 1024), mm_f(1024, rows, 3), 3, c);
        // the activation the fusion pass hands over after folding a batch into rows
        const ggml_tensor folded = htp_tensor_collapse_rows(&(const ggml_tensor &) mm_f(1024, rows, 3));
        emit(mm_w(t, 1024, 1024), folded, 3, c);
    }

    // 5. Solved by the batched HMX kernel, which the fused path has no counterpart for.
    for (int64_t rows : { 8, 33, 64 }) for (int64_t b : { 2, 4 }) for (int64_t g : { 1, 2 }) for (size_t vt : { 8u << 20, 1u << 20 }) {
        emit(mm_w(GGML_TYPE_F16, 256, 256, b, false), mm_f(256, rows, b * g), 2, { 8, vt, 1, 1 });
    }

    fprintf(stderr, "%d cases\n", n_cases);
    return 0;
}
