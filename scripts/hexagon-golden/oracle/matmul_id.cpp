// Case table for MUL_MAT_ID (routed experts) through the host's op-node path. Line format:
//   <weight> | <act> | <ids> | <dst> | <prec> | <threads> <vtcm> <hmx> <mdev> => <params>
// where <prec> is the op's `GGML_PREC_F32` request (op_params[3]) and every tensor is
// "type ne0..3 nb0..3 flags". The weight is [k, n, n_expert], the activations [k, ne11, tokens],
// the ids [n_used, tokens] (I32) and the output [n, n_used, tokens].
#include "mm_common.h"

static int n_cases = 0;

static ggml_tensor ids_t(ggml_type t, int64_t n_used, int64_t tokens) {
    const int64_t ie[4] = { n_used, tokens, 1, 1 };
    return oracle_tensor(t, ie);
}

static void emit(const ggml_tensor & w, const ggml_tensor & x, const ggml_tensor & ids, const ggml_tensor & d,
                 int prec, const MmCfg & c) {
    ggml_hexagon_session sess = mm_sess(c);
    ggml_tensor w0 = w, x0 = x, i0 = ids, dst = d;
    dst.op     = GGML_OP_MUL_MAT_ID;
    dst.src[0] = &w0;
    dst.src[1] = &x0;
    dst.src[2] = &i0;
    ((int32_t *) dst.op_params)[3] = prec;

    htp_opnode node(htp_op_remap(&dst), &dst);
    const bool ok = htp_kparams_solve(&sess, node);

    mm_print_t(w0); printf(" | "); mm_print_t(x0); printf(" | "); mm_print_t(i0); printf(" | "); mm_print_t(dst);
    printf(" | %d | ", prec); mm_print_cfg(c); printf(" =>");
    mm_print_words(ok, *node.kparams<htp_mm_kernel_params>());
    printf("\n");
    n_cases++;
}

// One routed matmul: `tokens` tokens, `n_used` of `n_exp` experts each; `ne11` activation rows per
// token (1 for gate/up, n_used for the down projection).
static void id(ggml_type wt, int64_t k, int64_t n, int64_t n_exp, int64_t n_used, int64_t tokens, int64_t ne11,
               const MmCfg & c, bool tiled = true, int prec = 0) {
    emit(mm_w(wt, k, n, n_exp, tiled), mm_f(k, ne11, tokens), ids_t(GGML_TYPE_I32, n_used, tokens),
         mm_f(n, n_used, tokens), prec, c);
}

int main() {
    opt_rt_repack = 0;
    const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K };

    // 1. Decode (one token) and prefill chunks at the shipped session.
    for (ggml_type t : types) for (int64_t k : { 256, 1024, 2048 }) for (int64_t n : { 32, 512, 1024, 2048 })
    for (int64_t n_exp : { 8, 32, 64 }) for (int64_t n_used : { 2, 4, 8 }) for (int64_t tokens : { 1, 2, 4, 5, 8, 33, 128 }) {
        if (n_used > n_exp) continue;
        id(t, k, n, n_exp, n_used, tokens, 1, MM_BASE);
        if (tokens == 1 || tokens == 8) id(t, k, n, n_exp, n_used, tokens, n_used, MM_BASE);  // down: one row per expert slot
    }

    // 2. Session sweep.
    for (ggml_type t : types) for (int64_t tokens : { 1, 3, 8, 33, 128 }) for (uint32_t th : { 1, 4, 8 })
    for (size_t vt : MM_VTCMS) for (uint32_t hmx : { 0u, 1u }) for (uint32_t mdev : { 1u, 2u }) {
        id(t, 1024, 1024, 32, 4, tokens, 1, { th, vt, hmx, mdev });
        id(t, 1024, 100, 32, 4, tokens, 1, { th, vt, hmx, mdev });   // N not a multiple of 32: no HMX
    }

    // 3. Other weight types and layouts.
    for (ggml_type t : { GGML_TYPE_Q4_1, GGML_TYPE_Q5_K, GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
                         GGML_TYPE_Q1_0, GGML_TYPE_Q2_0, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_BF16 })
    for (int64_t tokens : { 1, 8, 64 }) for (uint32_t hmx : { 0u, 1u }) for (bool tiled : { true, false }) {
        id(t, 1024, 512, 8, 2, tokens, 1, { 8, 8u << 20, hmx, 1 }, tiled);
    }

    // 4. Operands the host sends to the CPU.
    for (uint32_t hmx : { 0u, 1u }) {
        const MmCfg c = { 8, 8u << 20, hmx, 1 };
        const ggml_tensor ids = ids_t(GGML_TYPE_I32, 2, 8);
        id(GGML_TYPE_Q8_0, 1024, 512, 8, 2, 8, 1, c, true, GGML_PREC_F32);            // F32 precision requested
        emit(mm_w(GGML_TYPE_Q8_0, 1024, 512, 8), mm_f(1024, 1, 8), ids_t(GGML_TYPE_I64, 2, 8), mm_f(512, 2, 8), 0, c);
        emit(mm_w(GGML_TYPE_Q8_0, 1024, 512, 8), mm_f(1024, 1, 8, 0, GGML_TYPE_F16), ids, mm_f(512, 2, 8), 0, c);
        emit(mm_w(GGML_TYPE_Q8_0, 1024, 512, 8), mm_f(1024, 1, 8), ids, mm_f(512, 2, 8, 0, GGML_TYPE_F16), 0, c);
        {   // padded weight rows / permuted weight
            const int64_t we[4] = { 1024, 512, 8, 1 };
            emit(oracle_tensor(GGML_TYPE_Q8_0, we, 1, true), mm_f(1024, 1, 8), ids, mm_f(512, 2, 8), 0, c);
            emit(oracle_tensor(GGML_TYPE_Q8_0, we, 2, true), mm_f(1024, 1, 8), ids, mm_f(512, 2, 8), 0, c);
            emit(oracle_tensor(GGML_TYPE_F16, we, 1, false), mm_f(1024, 1, 8), ids, mm_f(512, 2, 8), 0, c);
        }
        id(GGML_TYPE_Q4_K, 1056, 512, 8, 2, 8, 1, c);                                  // width not a whole block
        id(GGML_TYPE_F16, 1040, 512, 8, 2, 8, 1, c, false);                            // F16 width not a multiple of 32
        // strided activations and output (slices of larger buffers)
        emit(mm_w(GGML_TYPE_Q8_0, 1024, 512, 8), mm_f(1024, 1, 8, 1), ids, mm_f(512, 2, 8, 1), 0, c);
        emit(mm_w(GGML_TYPE_Q8_0, 1024, 512, 8), mm_f(1024, 1, 8, 2), ids, mm_f(512, 2, 8, 2), 0, c);
    }

    fprintf(stderr, "%d cases\n", n_cases);
    return 0;
}
