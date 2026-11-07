// Case table for the fused routed matmuls (MUL_MAT_ID_NX): `htp_kparams_solve_mmidnx` on the first
// weight, the activation and the first output, then the fusion pass's kernel check
// (`is_supported_mul_mat_id_nx_kernel`, mirrored here). Line format:
//   <weight0> | <act> | <dst0> | <n_weights> | <threads> <vtcm> <hmx> <mdev> => <supported 0|1> <params> | unsupported
#include "mm_common.h"

static int n_cases = 0;

static void emit(const ggml_tensor & w, const ggml_tensor & x, const ggml_tensor & d, int n_weights, const MmCfg & c) {
    ggml_hexagon_session sess = mm_sess(c);
    ggml_tensor w0 = w, x0 = x, dst = d;
    dst.op = GGML_OP_MUL_MAT_ID;
    htp_mm_kernel_params kp;
    const bool ok = htp_kparams_solve_mmidnx(&sess, &w0, &x0, &dst, n_weights, &kp);

    mm_print_t(w0); printf(" | "); mm_print_t(x0); printf(" | "); mm_print_t(dst);
    printf(" | %d | ", n_weights); mm_print_cfg(c); printf(" =>");
    if (ok) {
        const bool tiled = ggml_hexagon_tensor_is_tiled(&w0);
        const bool supported = kp.n_hmx ? kp.kernel_type == HTP_MM_KERNEL_HMX_2D
                                        : tiled && (kp.kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW ||
                                                    kp.kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK);
        printf(" %d", supported ? 1 : 0);
    }
    mm_print_words(ok, kp);
    printf("\n");
    n_cases++;
}

static void id(ggml_type wt, int64_t k, int64_t n, int64_t n_exp, int64_t n_used, int64_t tokens, int64_t ne11,
               int nw, const MmCfg & c, bool tiled = true) {
    emit(mm_w(wt, k, n, n_exp, tiled), mm_f(k, ne11, tokens), mm_f(n, n_used, tokens), nw, c);
}

int main() {
    opt_rt_repack = 0;
    const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K };

    for (ggml_type t : types) for (int64_t k : { 256, 1024, 2048 }) for (int64_t n : { 32, 512, 2048 })
    for (int64_t n_exp : { 8, 64 }) for (int64_t n_used : { 2, 8 }) for (int64_t tokens : { 1, 4, 5, 8, 33, 128 })
    for (int nw : { 2, 3 }) {
        id(t, k, n, n_exp, n_used, tokens, 1, nw, MM_BASE);
    }
    for (ggml_type t : types) for (int64_t tokens : { 1, 8, 33 }) for (uint32_t th : { 1, 4, 8 }) for (size_t vt : MM_VTCMS)
    for (uint32_t hmx : { 0u, 1u }) for (uint32_t mdev : { 1u, 2u }) {
        id(t, 1024, 1024, 32, 4, tokens, 1, 2, { th, vt, hmx, mdev });
        id(t, 1024, 100, 32, 4, tokens, 1, 2, { th, vt, hmx, mdev });
    }
    for (ggml_type t : { GGML_TYPE_Q4_1, GGML_TYPE_Q5_K, GGML_TYPE_IQ4_NL, GGML_TYPE_F16 }) for (bool tiled : { true, false })
    for (int64_t tokens : { 1, 8 }) for (uint32_t hmx : { 0u, 1u }) {
        id(t, 1024, 512, 8, 2, tokens, 1, 2, { 8, 8u << 20, hmx, 1 }, tiled);
    }

    fprintf(stderr, "%d cases\n", n_cases);
    return 0;
}
