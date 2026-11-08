// Case table for the fused MUL_MAT+ADD (a matmul whose F32 bias the kernel adds):
// `htp_kparams_solve_matmul_add` on the matmul's operands, and the collapse the fused node then
// gets from `htp_opnode_collapse`. Line format:
//   <weight> | <act> | <bias | -> | <dst> | <threads> <vtcm> <hmx> <mdev> => <act'> | <dst'> | <bias' | -> | <params>
// where the primed tensors are the fused node's after the collapse ("=" when unchanged).
#include "mm_common.h"

static int n_cases = 0;

static void emit(const ggml_tensor & w, const ggml_tensor & x, const ggml_tensor * b, const ggml_tensor & d,
                 const MmCfg & c) {
    ggml_hexagon_session sess = mm_sess(c);
    ggml_tensor w0 = w, x0 = x, dst = d, bias;
    if (b) {
        bias = *b;
    }
    ggml_tensor mm = d;
    mm.op     = GGML_OP_MUL_MAT;
    mm.src[0] = &w0;
    mm.src[1] = &x0;
    dst.op     = GGML_OP_ADD;
    dst.src[0] = &mm;
    dst.src[1] = b ? &bias : nullptr;

    htp_mm_kernel_params kp;
    const bool ok = htp_kparams_solve_matmul_add(&sess, &w0, &x0, b ? &bias : nullptr, &dst, &kp);

    htp_opnode node(HTP_OP_MUL_MAT_ADD, &dst);
    node.inputs.clear();
    node.inputs.push_back(&w0);
    node.inputs.push_back(&x0);
    if (b) {
        node.inputs.push_back(&bias);
    }
    node.outputs.clear();
    node.outputs.push_back(&dst);
    htp_opnode_collapse(&sess, node);

    mm_print_t(w0); printf(" | "); mm_print_t(x0); printf(" | ");
    if (b) { mm_print_t(bias); } else { printf("-"); }
    printf(" | "); mm_print_t(dst); printf(" | "); mm_print_cfg(c); printf(" => ");
    mm_print_t_or_same(*node.inputs[1], x0); printf(" | "); mm_print_t_or_same(*node.outputs[0], dst); printf(" | ");
    if (b) { mm_print_t_or_same(*node.inputs[2], bias); } else { printf("-"); }
    printf(" |");
    mm_print_words(ok, kp);
    printf("\n");
    n_cases++;
}

int main() {
    opt_rt_repack = 0;
    const ggml_type F16 = GGML_TYPE_F16;

    struct W { ggml_type t; bool tiled; };
    const W weights[] = { { GGML_TYPE_Q8_0, true }, { GGML_TYPE_Q4_0, true }, { GGML_TYPE_Q4_K, true }, { F16, false } };

    // 1. A bias row or a full-shape bias over decode, prefill and tail row counts.
    for (const W & wt : weights) for (int64_t k : { 256, 1024 }) for (int64_t n : { 96, 1024, 3072 })
    for (int64_t m : { 1, 2, 5, 8, 33, 130 }) for (const MmCfg & c : {
            MM_BASE, MmCfg { 8, 8u << 20, 0, 1 }, MmCfg { 4, 1u << 20, 1, 1 }, MmCfg { 6, 128u << 10, 0, 1 },
            MmCfg { 8, 64u << 10, 1, 1 }, MmCfg { 8, 8u << 20, 1, 2 } }) {
        const ggml_tensor w = mm_w(wt.t, k, n, 1, wt.tiled);
        const ggml_tensor x = mm_f(k, m), d = mm_f(n, m);
        const ggml_tensor row = mm_f(n, 1), full = mm_f(n, m), padded = mm_f(n, m, 1, 1);
        emit(w, x, &row, d, c);
        emit(w, x, &full, d, c);
        emit(w, x, &padded, d, c);
        emit(w, x, nullptr, d, c);
    }

    // 2. Batched activations: the bias collapses with them or the node stays 4-D.
    for (const W & wt : weights) for (int64_t b : { 2, 3, 8 }) for (int64_t m : { 1, 5, 33 }) for (uint32_t hmx : { 0u, 1u }) {
        const MmCfg c = { 8, 8u << 20, hmx, 1 };
        const int64_t k = 256, n = 512;
        const ggml_tensor w = mm_w(wt.t, k, n, 1, wt.tiled);
        const ggml_tensor x = mm_f(k, m, b), d = mm_f(n, m, b);
        const ggml_tensor row = mm_f(n, 1), full = mm_f(n, m, b), per_batch = mm_f(n, 1, b), bad = mm_f(n, m, b, 3);
        emit(w, x, &row, d, c);
        emit(w, x, &full, d, c);
        emit(w, x, &per_batch, d, c);
        emit(w, x, &bad, d, c);          // planes interleaved: no collapse
        emit(w, mm_f(k, m, b, 3), &row, d, c);
        emit(w, x, nullptr, d, c);
    }

    fprintf(stderr, "%d cases\n", n_cases);
    return 0;
}
