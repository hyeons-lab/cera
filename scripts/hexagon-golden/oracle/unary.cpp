// Case table for the unary family (activations, SCALE/CLAMP, the norms, RMS_NORM_MUL and
// RMS_NORM_MUL_ADD, and the Snake/Sin/Cos/HardSigmoid/HardSwish/Elu extension ops) through the
// host's op-node path: `htp_opnode`, then `htp_kparams_solve(sess, node)`, which collapses
// elementwise nodes (`htp_opnode_collapse`) before solving. Line format:
//   <src0> | <src1 or -> | <dst> | <op name> <opcode> <session threads> <vtcm bytes> => <src0 after collapse> | <dst after collapse> | <words | unsupported>
// The collapsed tensors are what the real host registers with the DSP, so the line records them
// whether or not the solver then accepts the node.
#include "eltwise.h"

#include <array>
#include <vector>

static int n_cases = 0;

struct UOp {
    htp_op_code op;
    ggml_op     gop;   // GGML_OP_NONE: no ggml op maps to it, the node is built with the opcode directly
    int         uop;   // ggml_unary_op for GGML_OP_UNARY, else -1
};

static const UOp U_SILU        = { HTP_OP_UNARY_SILU, GGML_OP_UNARY, GGML_UNARY_OP_SILU };
static const UOp U_RMS         = { HTP_OP_RMS_NORM, GGML_OP_RMS_NORM, -1 };
static const UOp U_RMS_MUL     = { HTP_OP_RMS_NORM_MUL, GGML_OP_NONE, -1 };
static const UOp U_RMS_MUL_ADD = { HTP_OP_RMS_NORM_MUL_ADD, GGML_OP_NONE, -1 };
static const UOp U_SCALE       = { HTP_OP_SCALE, GGML_OP_SCALE, -1 };
static const UOp U_SQR         = { HTP_OP_SQR, GGML_OP_SQR, -1 };
static const UOp U_TRI         = { HTP_OP_TRI, GGML_OP_TRI, -1 };
static const UOp U_SNAKE       = { HTP_OP_UNARY_SNAKE, GGML_OP_NONE, -1 };

static const UOp ALL_OPS[] = {
    { HTP_OP_SCALE, GGML_OP_SCALE, -1 },
    { HTP_OP_CLAMP, GGML_OP_CLAMP, -1 },
    { HTP_OP_LEAKY_RELU, GGML_OP_LEAKY_RELU, -1 },
    { HTP_OP_SQR, GGML_OP_SQR, -1 },
    { HTP_OP_SQRT, GGML_OP_SQRT, -1 },
    { HTP_OP_UNARY_LOG, GGML_OP_LOG, -1 },
    { HTP_OP_NORM, GGML_OP_NORM, -1 },
    { HTP_OP_RMS_NORM, GGML_OP_RMS_NORM, -1 },
    { HTP_OP_L2_NORM, GGML_OP_L2_NORM, -1 },
    { HTP_OP_TRI, GGML_OP_TRI, -1 },
    { HTP_OP_UNARY_SILU, GGML_OP_UNARY, GGML_UNARY_OP_SILU },
    { HTP_OP_UNARY_GELU, GGML_OP_UNARY, GGML_UNARY_OP_GELU },
    { HTP_OP_UNARY_GELU_QUICK, GGML_OP_UNARY, GGML_UNARY_OP_GELU_QUICK },
    { HTP_OP_UNARY_GELU_ERF, GGML_OP_UNARY, GGML_UNARY_OP_GELU_ERF },
    { HTP_OP_UNARY_SIGMOID, GGML_OP_UNARY, GGML_UNARY_OP_SIGMOID },
    { HTP_OP_UNARY_NEG, GGML_OP_UNARY, GGML_UNARY_OP_NEG },
    { HTP_OP_UNARY_EXP, GGML_OP_UNARY, GGML_UNARY_OP_EXP },
    { HTP_OP_UNARY_SOFTPLUS, GGML_OP_UNARY, GGML_UNARY_OP_SOFTPLUS },
    { HTP_OP_UNARY_TANH, GGML_OP_UNARY, GGML_UNARY_OP_TANH },
    { HTP_OP_UNARY_ABS, GGML_OP_UNARY, GGML_UNARY_OP_ABS },
    { HTP_OP_UNARY_RELU, GGML_OP_UNARY, GGML_UNARY_OP_RELU },
    { HTP_OP_UNARY_STEP, GGML_OP_UNARY, GGML_UNARY_OP_STEP },
    { HTP_OP_UNARY_CEIL, GGML_OP_UNARY, GGML_UNARY_OP_CEIL },
    { HTP_OP_UNARY_SGN, GGML_OP_UNARY, GGML_UNARY_OP_SGN },
    { HTP_OP_UNARY_HARDSIGMOID, GGML_OP_UNARY, GGML_UNARY_OP_HARDSIGMOID },
    { HTP_OP_UNARY_HARDSWISH, GGML_OP_UNARY, GGML_UNARY_OP_HARDSWISH },
    { HTP_OP_UNARY_ELU, GGML_OP_UNARY, GGML_UNARY_OP_ELU },
    { HTP_OP_UNARY_SIN, GGML_OP_SIN, -1 },
    { HTP_OP_UNARY_COS, GGML_OP_COS, -1 },
    { HTP_OP_UNARY_SNAKE, GGML_OP_NONE, -1 },
    U_RMS_MUL,
    U_RMS_MUL_ADD,
};
static const int N_OPS = (int) (sizeof(ALL_OPS) / sizeof(ALL_OPS[0]));

static void emit(const UOp & u, const ggml_tensor & s0, const ggml_tensor * s1, const ggml_tensor & d,
                 uint32_t th, size_t vt) {
    ggml_hexagon_session sess;
    sess.n_threads = th;
    sess.vtcm_size = vt;

    ggml_tensor op = d;
    op.op     = u.gop;
    op.src[0] = const_cast<ggml_tensor *>(&s0);
    if (u.uop >= 0) {
        op.op_params[0] = u.uop;
    }

    const bool fused = u.op == HTP_OP_RMS_NORM_MUL || u.op == HTP_OP_RMS_NORM_MUL_ADD;
    const bool mapped = u.gop != GGML_OP_NONE;
    if (mapped && htp_op_remap(&op) != u.op) {
        fprintf(stderr, "oracle bug: op remap of %s\n", eltwise_op_name(u.op));
        abort();
    }

    // the node the host builds: the op's own tensors, or for a fused norm what fuse_rms_norm_mul
    // sets (input, weight, output)
    htp_opnode node(u.op, &op);
    if (fused) {
        node.inputs.clear();
        node.inputs.push_back(&s0);
        node.inputs.push_back(s1);
        node.outputs.clear();
        node.outputs.push_back(&d);
    }
    const bool ok = htp_kparams_solve(&sess, node);

    // the same node through the public entry points, which must agree
    if (mapped && !fused) {
        // the public entry copies a whole kernel_params blob (32 words) for a node-path op
        int32_t k2[HTP_OP_MAX_KERN_PARAMS] = {};
        const bool ok2 = htp_kparams_solve(&sess, &op, k2);
        if (ok != ok2 || (ok && memcmp(k2, node.kernel_params, sizeof(k2)) != 0)) {
            fprintf(stderr, "oracle bug: node path and htp_kparams_solve disagree for %s\n", eltwise_op_name(u.op));
            abort();
        }
    }
    if (fused) {
        htp_unary_kernel_params k2;
        memset(&k2, 0xAA, sizeof(k2));  // poison, as the other oracles do
        const bool ok2 = u.op == HTP_OP_RMS_NORM_MUL ? htp_kparams_solve_rms_norm_mul(&sess, &s0, s1, &d, &k2)
                                                      : htp_kparams_solve_rms_norm_mul_add(&sess, &s0, s1, &d, &k2);
        if (ok != ok2 || (ok && memcmp(&k2, node.kernel_params, sizeof(k2)) != 0)) {
            fprintf(stderr, "oracle bug: node path and the fused solver disagree for %s\n", eltwise_op_name(u.op));
            abort();
        }
    }

    oracle_print_tensor(s0); printf(" | "); print_tensor_or_dash(s1); printf(" | "); oracle_print_tensor(d);
    printf(" | %s %u %u %zu => ", eltwise_op_name(u.op), (unsigned) u.op, th, vt);
    oracle_print_tensor(*node.inputs[0]); printf(" | "); oracle_print_tensor(*node.outputs[0]); printf(" |");
    oracle_print_words(ok, node.kernel_params, sizeof(node.kernel_params));
    n_cases++;
}

// A same-shaped weight/addend for the fused norms; `kind` is its layout
static void emit_fused(const UOp & u, ggml_type t, int64_t w, int64_t r, int64_t p, int64_t b, int wkind,
                       int64_t ww, int64_t wr, int64_t wp, int64_t wb, int k0, int kd, uint32_t th, size_t vt) {
    ggml_tensor s0 = mk(t, w, r, p, b, k0), wt = mk(GGML_TYPE_F32, ww, wr, wp, wb, wkind), d = mk(t, w, r, p, b, kd);
    emit(u, s0, &wt, d, th, vt);
}

static void emit_plain(const UOp & u, ggml_type t, int64_t w, int64_t r, int64_t p, int64_t b, int k0, int kd,
                       uint32_t th, size_t vt) {
    ggml_tensor s0 = mk(t, w, r, p, b, k0), d = mk(t, w, r, p, b, kd);
    emit(u, s0, nullptr, d, th, vt);
}

static int rot = 0;
static const UOp & next_op() { return ALL_OPS[rot++ % N_OPS]; }

// dispatch to the plain or fused emitter with a weight shaped like the row broadcast cera uses
static void emit_any(const UOp & u, ggml_type t, int64_t w, int64_t r, int64_t p, int64_t b, uint32_t th, size_t vt) {
    if (u.op == HTP_OP_RMS_NORM_MUL || u.op == HTP_OP_RMS_NORM_MUL_ADD) {
        emit_fused(u, t, w, r, p, b, 0, w, 1, 1, 1, 0, 0, th, vt);
    } else {
        emit_plain(u, t, w, r, p, b, 0, 0, th, vt);
    }
}

struct Run { uint32_t th; size_t vt; };

typedef std::array<int64_t, 4> S4;

int main() {
    const ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16 };
    const int64_t widths[]  = { 1, 3, 4, 8, 16, 31, 32, 100, 128, 576, 1024, 2048, 4096, 8192, 65536 };
    const int64_t shapes[][3] = { { 1, 1, 1 }, { 2, 1, 1 }, { 3, 1, 1 }, { 7, 1, 1 }, { 17, 1, 1 }, { 64, 1, 1 },
                                  { 512, 1, 1 }, { 1000, 1, 1 }, { 4, 3, 1 }, { 5, 2, 2 }, { 9, 3, 2 }, { 2, 2, 2 } };
    const Run      lite[] = { { 6, 8u << 20 }, { 1, 8u << 20 }, { 4, 64u << 10 } };
    const uint32_t ths[]  = { 1, 2, 4, 6, 8 };
    const size_t   vts[]  = { 8u << 20, 1u << 20, 64u << 10, 0 };

    // 1. the shape grid with a rotating op: kernel selection does not depend on the op except
    //    through the weight (fused norms), the F16 allow-list and TRI, which later blocks pin
    for (ggml_type t : types) for (int64_t w : widths) for (auto & s : shapes) for (const Run & ru : lite)
        emit_any(next_op(), t, w, s[0], s[1], s[2], ru.th, ru.vt);

    // 2. every op on a spread of shapes, both types, the full thread and budget grid on a few
    for (int oi = 0; oi < N_OPS; oi++) for (ggml_type t : types)
        for (const S4 & sh : std::vector<S4>{ { 8192, 1, 1, 1 }, { 576, 7, 1, 1 }, { 16, 100, 1, 1 }, { 100, 3, 2, 2 }, { 4096, 3, 1, 1 } })
            for (const Run & ru : lite)
                emit_any(ALL_OPS[oi], t, sh[0], sh[1], sh[2], sh[3], ru.th, ru.vt);
    for (const UOp & u : std::vector<UOp>{ U_SILU, U_RMS, U_RMS_MUL })
        for (const S4 & sh : std::vector<S4>{ { 8192, 1, 1, 1 }, { 100, 17, 1, 1 }, { 16, 1000, 1, 1 }, { 4096, 3, 1, 1 }, { 65536, 1, 1, 1 } })
            for (uint32_t th : ths) for (size_t vt : vts)
                emit_any(u, GGML_TYPE_F32, sh[0], sh[1], sh[2], sh[3], th, vt);

    // 3. the three collapse steps: merging rows into columns, merging in chunks of k rows, and
    //    splitting a long row, at every thread count and in both element widths
    {
        const int64_t cs[][4] = { { 8, 64, 1, 1 }, { 4, 3, 5, 2 }, { 1, 7, 3, 2 }, { 16, 1, 5, 3 }, { 1, 1, 1, 1 },
                                  { 31, 66, 1, 1 }, { 16, 1000, 1, 1 }, { 30, 77, 2, 1 }, { 30, 64, 3, 1 }, { 3, 1024, 1, 1 },
                                  { 64, 33, 1, 1 }, { 64, 32, 1, 1 }, { 64, 31, 1, 1 }, { 32, 65, 1, 1 }, { 127, 10, 1, 1 },
                                  { 8192, 1, 1, 1 }, { 4096, 2, 1, 1 }, { 2048, 5, 1, 1 }, { 16384, 3, 1, 1 }, { 1024, 1, 1, 1 },
                                  { 1023, 1, 1, 1 }, { 4100, 1, 1, 1 }, { 4096, 1, 1, 1 }, { 12288, 1, 1, 1 }, { 6000, 1, 1, 1 },
                                  { 65536, 1, 1, 1 }, { 8192, 1, 2, 1 }, { 8192, 2, 2, 1 }, { 5000, 1, 3, 1 }, { 2048, 1, 1, 7 },
                                  { 20000, 1, 1, 1 }, { 192, 3, 1, 1 }, { 96, 5, 1, 1 }, { 4, 512, 1, 1 }, { 4, 2048, 1, 1 },
                                  { 4, 4096, 1, 1 }, { 33, 63, 1, 1 }, { 24, 85, 1, 1 }, { 2, 1024, 2, 2 }, { 2, 4, 1024, 1 } };
        const UOp ops[] = { U_SILU, U_SQR, U_SNAKE };
        for (ggml_type t : types) for (const UOp & u : ops) for (auto & c : cs) for (uint32_t th : ths)
            emit_plain(u, t, c[0], c[1], c[2], c[3], 0, 0, th, 8u << 20);
        // the same with padded rows and planes, which break the row-contiguity the merges need
        for (int kind : { 1, 3, 4, 5 }) for (auto & c : cs) for (uint32_t th : { 1, 6 })
            emit_plain(U_SILU, GGML_TYPE_F32, c[0], c[1], c[2], c[3], kind, 0, th, 8u << 20);
        // small budgets decide whether the collapsed rows fit
        for (auto & c : cs) for (size_t vt : { 1u << 20, 64u << 10, 16u << 10 }) for (uint32_t th : { 1, 8 })
            emit_plain(U_SILU, GGML_TYPE_F32, c[0], c[1], c[2], c[3], 0, 0, th, vt);
    }

    // 4. layouts: every layout on the source, and on the destination (which must be contiguous)
    for (const UOp & u : std::vector<UOp>{ U_SILU, U_RMS, U_SCALE, U_TRI })
        for (const S4 & sh : std::vector<S4>{ { 96, 5, 1, 1 }, { 96, 4, 3, 1 }, { 100, 3, 2, 2 }, { 8, 64, 1, 1 }, { 1024, 4, 3, 1 } })
            for (int k0 : { 0, 1, 2, 3, 4, 5 }) for (int kd : { 0, 1, 2, 3 }) for (ggml_type t : types)
                emit_plain(u, t, sh[0], sh[1], sh[2], sh[3], k0, kd, 6, 8u << 20);

    // 5. mismatched or unsupported tensors
    {
        ggml_tensor f32 = mk(GGML_TYPE_F32, 64, 4, 1, 1), f16 = mk(GGML_TYPE_F16, 64, 4, 1, 1);
        ggml_tensor i32 = mk(GGML_TYPE_I32, 64, 4, 1, 1);
        ggml_tensor big = mk(GGML_TYPE_F32, 64, 8, 1, 1), wide = mk(GGML_TYPE_F32, 128, 4, 1, 1);
        for (const UOp & u : std::vector<UOp>{ U_SILU, U_RMS, U_SCALE }) {
            emit(u, f32, nullptr, f16, 6, 8u << 20);   // dst type differs
            emit(u, f16, nullptr, f32, 6, 8u << 20);
            emit(u, i32, nullptr, i32, 6, 8u << 20);
            emit(u, f32, nullptr, big, 6, 8u << 20);   // dst shape differs
            emit(u, f32, nullptr, wide, 6, 8u << 20);
        }
    }

    // 6. fused norms: weight broadcast or same-shaped, contiguous or padded, mismatched widths, no
    //    weight room (budget), the residual variant that needs a broadcast weight
    for (const UOp & u : std::vector<UOp>{ U_RMS_MUL, U_RMS_MUL_ADD }) {
        for (const S4 & sh : std::vector<S4>{ { 64, 1, 1, 1 }, { 576, 7, 1, 1 }, { 1024, 32, 1, 1 }, { 2048, 3, 2, 1 }, { 100, 5, 1, 1 }, { 8192, 4, 1, 1 } }) {
            const int64_t w = sh[0], r = sh[1], p = sh[2], b = sh[3];
            for (const Run & ru : std::vector<Run>{ { 6, 8u << 20 }, { 1, 8u << 20 }, { 8, 1u << 20 }, { 4, 64u << 10 }, { 6, 0 } }) {
                emit_fused(u, GGML_TYPE_F32, w, r, p, b, 0, w, 1, 1, 1, 0, 0, ru.th, ru.vt);      // broadcast row
                emit_fused(u, GGML_TYPE_F32, w, r, p, b, 0, w, r, p, b, 0, 0, ru.th, ru.vt);      // same shape
                emit_fused(u, GGML_TYPE_F32, w, r, p, b, 1, w, r, p, b, 0, 0, ru.th, ru.vt);      // same shape, padded rows
                emit_fused(u, GGML_TYPE_F32, w, r, p, b, 4, w, r, p, b, 0, 0, ru.th, ru.vt);      // same shape, strided rows
                emit_fused(u, GGML_TYPE_F32, w, r, p, b, 0, w / 2 ? w / 2 : 1, 1, 1, 1, 0, 0, ru.th, ru.vt);   // narrower weight
                emit_fused(u, GGML_TYPE_F32, w, r, p, b, 0, w * 2, 1, 1, 1, 0, 0, ru.th, ru.vt);               // wider weight
                emit_fused(u, GGML_TYPE_F32, w, r, p, b, 0, w, 1, 1, 1, 1, 0, ru.th, ru.vt);      // padded source rows
                emit_fused(u, GGML_TYPE_F32, w, r, p, b, 0, w, 1, 1, 1, 0, 1, ru.th, ru.vt);      // padded destination rows
                emit_fused(u, GGML_TYPE_F16, w, r, p, b, 0, w, 1, 1, 1, 0, 0, ru.th, ru.vt);      // F16 activations
            }
            emit_fused(u, GGML_TYPE_F32, w, r, p, b, 0, w, r > 1 ? r - 1 : 2, 1, 1, 0, 0, 6, 8u << 20);   // weight rows differ
            emit_fused(u, GGML_TYPE_F32, w, r, p, b, 2, w, r, p, b, 2, 0, 6, 8u << 20);       // transposed
        }
    }

    // 7. every op once more on the cera shapes: a decode vector, a prefill chunk, a QK-norm batch
    for (int oi = 0; oi < N_OPS; oi++)
        for (const S4 & sh : std::vector<S4>{ { 2048, 1, 1, 1 }, { 2048, 512, 1, 1 }, { 64, 32, 1, 1 }, { 64, 8192, 1, 1 }, { 128, 16, 4, 1 } })
            for (const Run & ru : std::vector<Run>{ { 6, 8u << 20 }, { 8, 8u << 20 } })
                emit_any(ALL_OPS[oi], GGML_TYPE_F32, sh[0], sh[1], sh[2], sh[3], ru.th, ru.vt);

    fprintf(stderr, "%d cases\n", n_cases);
    return 0;
}
