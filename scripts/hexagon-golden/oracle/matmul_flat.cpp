// Case table for MUL_MAT with plain (untiled) F16 / F32 / BF16 weights: the HMX flat and batched
// kernels, the HVX flat kernels, and the collapse of batched activations. Same line format as
// matmul.cpp:
//   <weight> | <act> | <dst> | <threads> <vtcm> <hmx> <mdev> => <act'> | <dst'> | <params> | <raw>
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
        ggml_hexagon_session big = sess;
        big.vtcm_size = 256u << 20;
        htp_mm_kernel_params raw;
        const bool types_ok = htp_kparams_solve(&big, &dst, nullptr);
        const bool raw_ok   = types_ok && htp_kparams_solve_mmidnx(&sess, &w0, &x0, &dst, 0, &raw);
        if (!raw_ok) {
            memset(&raw, 0, sizeof(raw));
        }
        printf("raw");
        mm_print_words(raw_ok, raw);
    } else {
        printf("=");
    }
    printf("\n");
    n_cases++;
}

static ggml_tensor pw(ggml_type t, int64_t k, int64_t n, int64_t ne2 = 1, int kind = 0) {
    const int64_t ne[4] = { k, n, ne2, 1 };
    return oracle_tensor(t, ne, kind, false);
}

int main() {
    opt_rt_repack = 0;
    const ggml_type F32 = GGML_TYPE_F32, F16 = GGML_TYPE_F16, BF16 = GGML_TYPE_BF16;

    // 1. Plain F16 weights against F32 activations (the decision model's encoder, the ViT
    //    projections, Whisper): every row count from decode to a long prefill chunk.
    {
        const int64_t ks[] = { 32, 64, 96, 128, 512, 768, 1024, 3072, 4096 };
        const int64_t ns[] = { 32, 64, 96, 512, 1024, 3072, 4608 };
        const int64_t ms[] = { 1, 4, 5, 7, 8, 16, 32, 33, 64, 200, 256, 1000 };
        for (int64_t k : ks) for (int64_t n : ns) for (int64_t m : ms) {
            emit(pw(F16, k, n), mm_f(k, m), mm_f(n, m), MM_BASE);
        }
    }
    {
        const int64_t shapes[][2] = { { 1024, 1024 }, { 768, 3072 }, { 3072, 768 } };
        for (auto & s : shapes) for (int64_t m : { 1, 5, 8, 33, 200 }) for (uint32_t th : { 1, 4, 8 })
        for (size_t vt : MM_VTCMS) for (uint32_t hmx : { 0u, 1u }) {
            emit(pw(F16, s[0], s[1]), mm_f(s[0], m), mm_f(s[1], m), { th, vt, hmx, 1 });
        }
    }
    // F16 activations, padded / transposed / sliced operands, odd widths, two devices
    for (int64_t m : { 1, 8, 33, 200 }) for (uint32_t hmx : { 0u, 1u }) for (size_t vt : { 8u << 20, 1u << 20, 64u << 10 }) {
        const MmCfg c = { 8, vt, hmx, 1 };
        emit(pw(F16, 1024, 1024), mm_f(1024, m, 1, 0, F16), mm_f(1024, m), c);
        emit(pw(F16, 768, 3072), mm_f(768, m, 1, 1), mm_f(3072, m), c);
        emit(pw(F16, 768, 3072), mm_f(768, m), mm_f(3072, m, 1, 1), c);
        emit(pw(F16, 768, 3072), mm_f(768, m, 1, 2), mm_f(3072, m), c);
        emit(pw(F16, 768, 3072), mm_f(768, m), mm_f(3072, m, 1, 2), c);
        emit(pw(F16, 768, 3072, 1, 1), mm_f(768, m), mm_f(3072, m), c);   // padded weight rows
        emit(pw(F16, 1000, 96), mm_f(1000, m), mm_f(96, m), c);           // K not a multiple of 32
        emit(pw(F16, 1024, 1024), mm_f(1024, m), mm_f(1024, m), { 8, vt, hmx, 2 });
        emit(pw(F16, 768, 3072), mm_f(768, m, 1, 1), mm_f(3072, m, 1, 1), { 8, vt, hmx, 2 });
    }

    // 2. F32 weights and activations (attention scores and values on the HVX), plain and batched
    //    over heads; K-cache style views with padded rows or the heads interleaved.
    {
        const int64_t ks[] = { 64, 128, 384, 1024 };
        const int64_t ns[] = { 1, 13, 126, 1500 };
        const int64_t hs[] = { 1, 6, 8, 16 };
        const int64_t ms[] = { 1, 5, 126, 1500 };
        for (int64_t k : ks) for (int64_t n : ns) for (int64_t h : hs) for (int64_t m : ms)
        for (uint32_t hmx : { 0u, 1u }) for (size_t vt : { 8u << 20, 1u << 20, 128u << 10 }) {
            if ((k > 384 || n > 126) && h > 8) continue;
            emit(pw(F32, k, n, h), mm_f(k, m, h), mm_f(n, m, h), { 8, vt, hmx, 1 });
            if (hmx == 0 && vt == (8u << 20)) {
                emit(pw(F32, k, n, h, 1), mm_f(k, m, h), mm_f(n, m, h), MM_BASE);   // padded rows
                emit(pw(F32, k, n, h, 3), mm_f(k, m, h), mm_f(n, m, h), MM_BASE);   // heads interleaved
                emit(pw(F32, k, n, h), mm_f(k, m, h, 2), mm_f(n, m, h), MM_BASE);   // transposed activations
                emit(pw(F32, k, n, h), mm_f(k, m, h), mm_f(n, m, h, 2), MM_BASE);   // transposed output
            }
        }
    }
    // BF16 weights
    for (int64_t m : { 1, 5, 64 }) for (uint32_t hmx : { 0u, 1u }) for (size_t vt : { 8u << 20, 128u << 10 })
    for (int64_t k : { 64, 1024 }) for (int64_t n : { 96, 1024 }) {
        emit(pw(BF16, k, n), mm_f(k, m), mm_f(n, m), { 8, vt, hmx, 1 });
        emit(pw(BF16, k, n, 4), mm_f(k, m, 4), mm_f(n, m, 4), { 8, vt, hmx, 1 });
    }

    // 3. Batched F16 weights: grouped (GQA, several activation batches per weight batch) and
    //    one-to-one, which reach the batched HMX kernel (the solver only groups when it must).
    for (int64_t ne02 : { 2, 4, 8 }) for (int64_t g : { 1, 2, 4, 8 }) for (int64_t k : { 64, 128 })
    for (int64_t n : { 32, 64, 126, 512 }) for (int64_t m : { 1, 5, 8, 32, 64, 128 })
    for (uint32_t hmx : { 0u, 1u }) for (size_t vt : { 8u << 20, 1u << 20, 256u << 10 }) {
        const int64_t ne12 = ne02 * g;
        if (vt != (8u << 20) && (g > 2 || ne02 > 4)) continue;
        emit(pw(F16, k, n, ne02), mm_f(k, m, ne12), mm_f(n, m, ne12), { 8, vt, hmx, 1 });
        if (hmx == 1 && vt == (8u << 20)) {
            emit(pw(F16, k, n, ne02, 1), mm_f(k, m, ne12), mm_f(n, m, ne12), MM_BASE);
            emit(pw(F16, k, n, ne02), mm_f(k, m, ne12, 0, F16), mm_f(n, m, ne12), MM_BASE);
            emit(pw(F16, k, n, ne02), mm_f(k, m, ne12, 3), mm_f(n, m, ne12), MM_BASE);
            emit(pw(F16, k, n, ne02), mm_f(k, m, ne12), mm_f(n, m, ne12), { 8, vt, hmx, 2 });
            emit(pw(F16, k, n, ne02), mm_f(k, m, ne12), mm_f(n, m, ne12), { 3, vt, hmx, 1 });
        }
    }

    // 4. A 2-D weight against batched activations folds into rows when the batch is a plain repeat.
    for (ggml_type t : { F16, F32 }) for (int64_t b : { 2, 3, 8 }) for (int64_t m : { 1, 3, 5, 33, 130 })
    for (uint32_t hmx : { 0u, 1u }) for (size_t vt : { 8u << 20, 1u << 20, 128u << 10 }) {
        const MmCfg c = { 8, vt, hmx, 1 };
        emit(pw(t, 256, 1024), mm_f(256, m, b), mm_f(1024, m, b), c);
        emit(pw(t, 256, 1024), mm_f(256, m, b, 1), mm_f(1024, m, b, 1), c);
        emit(pw(t, 256, 1024), mm_f(256, m, b, 3), mm_f(1024, m, b), c);
        emit(pw(t, 256, 1024), mm_f(256, m, b, 2), mm_f(1024, m, b), c);
    }

    // 5. Operands the host sends to the CPU.
    for (uint32_t hmx : { 0u, 1u }) {
        const MmCfg c = { 8, 8u << 20, hmx, 1 };
        emit(pw(F32, 256, 256), mm_f(256, 8, 1, 0, F16), mm_f(256, 8), c);                       // F32 weights need F32 activations
        emit(pw(F16, 256, 256), mm_f(256, 8), mm_f(256, 8, 1, 0, F16), c);                       // F16 output
        emit(pw(F16, 256, 256), mm_f(256, 8, 1, 0, GGML_TYPE_I32), mm_f(256, 8), c);
        emit(pw(F16, 256, 256, 1, 2), mm_f(256, 8), mm_f(256, 8), c);                            // transposed weight view
        emit(pw(F32, 256, 256, 1, 2), mm_f(256, 8), mm_f(256, 8), c);
        emit(pw(F16, 256, 256, 4), mm_f(256, 8, 3), mm_f(256, 8, 3), c);                         // 3 % 4
        emit(pw(F16, 256, 256, 4), mm_f(256, 8, 2), mm_f(256, 8, 2), c);                         // fewer batches than weights
        emit(pw(GGML_TYPE_Q8_0, 256, 256, 1, false), mm_f(256, 8), mm_f(256, 8), c);             // quantized, not tiled
        emit(pw(GGML_TYPE_Q4_0, 256, 256, 1, false), mm_f(256, 8), mm_f(256, 8), c);
        emit(pw(GGML_TYPE_I32, 256, 256), mm_f(256, 8), mm_f(256, 8), c);                        // not a weight type
    }

    fprintf(stderr, "%d cases\n", n_cases);
    return 0;
}
