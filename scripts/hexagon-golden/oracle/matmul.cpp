// Case table for MUL_MAT with tiled quantized weights (the layouts cera uploads) through the
// host's op-node path: `htp_opnode_collapse` (a 2-D weight against batched activations folds them
// into rows), then `htp_kparams_solve(sess, node)`. Line format:
//   <weight> | <act> | <dst> | <threads> <vtcm> <hmx> <mdev> => <act'> | <dst'> | <params> | <raw>
// where act' / dst' are the node's tensors after the collapse ("=" when unchanged), <params> are
// the 23 words of `htp_mm_kernel_params` (or "unsupported"), and <raw> the params of the same op
// solved against the tensors as given, without collapsing ("=" unless the collapse applies).
// Every tensor is "type ne0..3 nb0..3 flags".
#include "mm_common.h"

static int n_cases = 0;

static void emit(const ggml_tensor & w, const ggml_tensor & x, const ggml_tensor & d, const MmCfg & c) {
    ggml_hexagon_session sess = mm_sess(c);
    ggml_tensor w0 = w, x0 = x, dst = d;
    dst.op     = GGML_OP_MUL_MAT;
    dst.src[0] = &w0;
    dst.src[1] = &x0;

    htp_opnode node(htp_op_remap(&dst), &dst);
    const bool ok = htp_kparams_solve(&sess, node);
    const ggml_tensor * xc = node.inputs[1];
    const ggml_tensor * dc = node.outputs[0];

    mm_print_t(w0); printf(" | "); mm_print_t(x0); printf(" | "); mm_print_t(dst);
    printf(" | "); mm_print_cfg(c); printf(" => ");
    mm_print_t_or_same(*xc, x0); printf(" | "); mm_print_t_or_same(*dc, dst); printf(" |");
    mm_print_words(ok, *node.kparams<htp_mm_kernel_params>());
    printf(" | ");
    if (htp_matmul_can_collapse(&w0, &x0, &dst)) {
        // The check that the operand types are fine at all is a separate static function: ask
        // the op path again with room to spare, then solve the original tensors directly.
        ggml_hexagon_session big = sess;
        big.vtcm_size = 256u << 20;
        htp_mm_kernel_params raw;
        const bool types_ok = htp_kparams_solve(&big, &dst, nullptr);
        const bool raw_ok   = types_ok && htp_kparams_solve_mmidnx(&sess, &w0, &x0, &dst, 0, &raw);
        if (!raw_ok) {
            memset(&raw, 0, sizeof(raw));
        }
        // a leading "raw" keeps an unsupported raw solve distinct from the "=" placeholder
        printf("raw");
        mm_print_words(raw_ok, raw);
    } else {
        printf("=");
    }
    printf("\n");
    n_cases++;
}

static void mm(ggml_type wt, int64_t k, int64_t n, int64_t m, const MmCfg & c = MM_BASE) {
    emit(mm_w(wt, k, n), mm_f(k, m), mm_f(n, m), c);
}

int main() {
    opt_rt_repack = 0;

    const ggml_type main_types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K };
    auto k_quant = [](ggml_type t) { return t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q6_K; };

    // 1. The shapes cera runs (LFM2 hidden 1024/2048, FFN 4608-8192, the 65536-row head, ViT and
    //    audio widths) over the activation row counts of decode, prefill chunks and tails, at the
    //    shipped session.
    {
        const int64_t ks_plain[] = { 32, 64, 96, 256, 512, 1024, 2048, 4608, 8192 };
        const int64_t ks_k[]     = { 256, 512, 768, 1024, 2048, 4608, 8192 };
        const int64_t ns[]       = { 32, 96, 512, 1024, 2048, 4608, 8192, 65536 };
        const int64_t ms[]       = { 1, 2, 4, 5, 8, 13, 32, 33, 64, 130, 512 };
        for (ggml_type t : main_types) {
            const bool kq = k_quant(t);
            for (int64_t k : kq ? std::vector<int64_t>(ks_k, ks_k + 7) : std::vector<int64_t>(ks_plain, ks_plain + 9))
            for (int64_t n : ns) for (int64_t m : ms) {
                mm(t, k, n, m);
            }
        }
    }

    // 2. Session sweep: thread counts, VTCM budgets from roomy to none, HMX on and off.
    {
        const int64_t shapes[][2] = { { 1024, 1024 }, { 1024, 4608 }, { 4608, 1024 }, { 2048, 8192 } };
        const int64_t ms[]        = { 1, 5, 33, 130, 512 };
        const uint32_t ths[]      = { 1, 4, 6 };
        for (ggml_type t : main_types) for (auto & s : shapes) for (int64_t m : ms)
        for (uint32_t th : ths) for (size_t vt : MM_VTCMS) for (uint32_t hmx : { 0u, 1u }) {
            mm(t, s[0], s[1], m, { th, vt, hmx, 1 });
        }
    }

    // 3. The other tiled types: one K per block size and a few shapes.
    {
        const ggml_type others[] = { GGML_TYPE_Q4_1, GGML_TYPE_Q5_K, GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4,
                                     GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q1_0, GGML_TYPE_Q2_0 };
        for (ggml_type t : others) for (int64_t k : { 1024, 2048 }) for (int64_t n : { 96, 1024, 4608 })
        for (int64_t m : { 1, 3, 5, 64, 130 }) for (size_t vt : { 8u << 20, 1u << 20, 64u << 10 }) {
            mm(t, k, n, m, { 6, vt, 1, 1 });
        }
    }

    // 4. Operand variants of the common weights: unpadded activation / output widths (K and N not a
    //    multiple of 32 on the model side), padded rows, F16 activations, transposed views.
    for (ggml_type t : { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K }) for (int64_t m : { 1, 5, 33, 130 })
    for (uint32_t hmx : { 0u, 1u }) {
        const MmCfg c = { 8, 8u << 20, hmx, 1 };
        const int64_t k = 1024, n = 2048;
        // activation narrower than the padded weight (in_dim 1000 -> 1024), output narrower (out_dim 2000 -> 2048)
        emit(mm_w(t, k, n), mm_f(1000, m), mm_f(n, m), c);
        emit(mm_w(t, k, n), mm_f(k, m), mm_f(2000, m), c);
        // padded activation and output rows (slices of wider buffers)
        emit(mm_w(t, k, n), mm_f(k, m, 1, 1), mm_f(n, m), c);
        emit(mm_w(t, k, n), mm_f(k, m), mm_f(n, m, 1, 1), c);
        // F16 activations
        emit(mm_w(t, k, n), mm_f(k, m, 1, 0, GGML_TYPE_F16), mm_f(n, m), c);
        // transposed activation or output view
        emit(mm_w(t, k, n), mm_f(k, m, 1, 2), mm_f(n, m), c);
        emit(mm_w(t, k, n), mm_f(k, m), mm_f(n, m, 1, 2), c);
        // two devices
        emit(mm_w(t, k, n), mm_f(k, m), mm_f(n, m), { 8, 8u << 20, hmx, 2 });
        emit(mm_w(t, k, n), mm_f(k, m, 1, 1), mm_f(n, m, 1, 1), { 8, 8u << 20, hmx, 2 });
        emit(mm_w(t, 1000 + 24, 96), mm_f(1000, m), mm_f(90, m), c);
    }

    // 5. Batched activations: a 2-D weight against [k, m, batch] folds into rows when the batch is
    //    a plain repeat, otherwise runs as the 4-D kernel; a batched weight does neither.
    for (ggml_type t : { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K }) for (int64_t m : { 1, 3, 5, 33 })
    for (int64_t b : { 2, 3, 8 }) for (uint32_t hmx : { 0u, 1u }) for (size_t vt : { 8u << 20, 1u << 20, 128u << 10 }) {
        const MmCfg c = { 8, vt, hmx, 1 };
        const int64_t k = 1024, n = 1024;
        emit(mm_w(t, k, n), mm_f(k, m, b), mm_f(n, m, b), c);
        emit(mm_w(t, k, n), mm_f(k, m, b, 1), mm_f(n, m, b, 1), c);   // row-padded: still folds (rows are a plain run)
        emit(mm_w(t, k, n), mm_f(k, m, b, 3), mm_f(n, m, b), c);       // planes interleaved: cannot fold
        emit(mm_w(t, k, n, 2), mm_f(k, m, 2 * b), mm_f(n, m, 2 * b), c);  // batched weight
        emit(mm_w(t, k, n, b), mm_f(k, m, b), mm_f(n, m, b), c);
    }

    // 6. Weights the DSP's tiled layout also holds: F16 (and F32 / BF16 stored as F16).
    for (ggml_type t : { GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_BF16 }) for (int64_t m : { 1, 5, 64 })
    for (uint32_t hmx : { 0u, 1u }) for (size_t vt : { 8u << 20, 1u << 20 }) {
        for (int64_t k : { 64, 1024 }) for (int64_t n : { 96, 1024 }) {
            mm(t, k, n, m, { 8, vt, hmx, 1 });
        }
    }

    // 7. Operands the host sends to the CPU: wrong output or activation type, a weight that is not
    //    a whole number of blocks wide, one that is not contiguous or is permuted, activations whose
    //    batch does not divide, quantized weights that are not tiled.
    for (uint32_t hmx : { 0u, 1u }) {
        const MmCfg c = { 8, 8u << 20, hmx, 1 };
        const ggml_type t = GGML_TYPE_Q8_0;
        emit(mm_w(t, 1024, 1024), mm_f(1024, 8), mm_f(1024, 8, 1, 0, GGML_TYPE_F16), c);
        emit(mm_w(t, 1024, 1024), mm_f(1024, 8, 1, 0, GGML_TYPE_I32), mm_f(1024, 8), c);
        emit(mm_w(GGML_TYPE_Q4_K, 1056, 1024), mm_f(1056, 8), mm_f(1024, 8), c);  // 1056 = 4.1 blocks of 256
        emit(mm_w(t, 1040, 1024), mm_f(1040, 8), mm_f(1024, 8), c);                // not a block of 32
        emit(mm_w(t, 1024, 1024, 1, false), mm_f(1024, 8), mm_f(1024, 8), c);      // quantized, not tiled
        { const int64_t ne[4] = { 1024, 1024, 1, 1 }; emit(oracle_tensor(t, ne, 1, true), mm_f(1024, 8), mm_f(1024, 8), c); }  // padded rows
        { const int64_t ne[4] = { 1024, 1024, 1, 1 }; emit(oracle_tensor(t, ne, 2, true), mm_f(1024, 8), mm_f(1024, 8), c); }  // permuted
        emit(mm_w(t, 1024, 1024, 2), mm_f(1024, 8, 3), mm_f(1024, 8, 3), c);       // 3 % 2 != 0
        emit(mm_w(t, 1024, 1024, 4), mm_f(1024, 8, 2), mm_f(1024, 8, 2), c);       // fewer batches than weights
    }

    fprintf(stderr, "%d cases\n", n_cases);
    return 0;
}
