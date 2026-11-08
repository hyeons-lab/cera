// Case table for the GLU family through the host's op-node path (`htp_opnode`, then
// `htp_kparams_solve(sess, node)`, which collapses a two-source node before solving), and through
// the exported `htp_kparams_solve_unary_mul_glu` directly. Line format:
//   <src0> | <src1 or -> | <dst> | <op name> <opcode> <node|direct> <session threads> <vtcm bytes> => <src0 after collapse> | <src1 after collapse or -> | <dst after collapse> | <words | unsupported>
// A node with no src1 is the single-source form: src0 holds both halves of each row and dst is
// half as wide. It is never collapsed.
#include "eltwise.h"

#include <array>
#include <vector>

static int n_cases = 0;

struct GOp {
    htp_op_code op;
    int         glu;   // ggml_glu_op, or -1 when ggml has no GLU op for it
};

static const GOp GLU_OPS[] = {
    { HTP_OP_GLU_SWIGLU, GGML_GLU_OP_SWIGLU },
    { HTP_OP_GLU_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_OAI },
    { HTP_OP_GLU_SWIGLU_CLAMP, GGML_GLU_OP_SWIGLU_CLAMP },
    { HTP_OP_GLU_GEGLU, GGML_GLU_OP_GEGLU },
    { HTP_OP_GLU_GEGLU_QUICK, GGML_GLU_OP_GEGLU_QUICK },
    { HTP_OP_GLU_GEGLU_ERF, GGML_GLU_OP_GEGLU_ERF },
    { HTP_OP_GLU_REGLU, GGML_GLU_OP_REGLU },
    { HTP_OP_GLU_SIGMOID, -1 },
};
static const int N_OPS = (int) (sizeof(GLU_OPS) / sizeof(GLU_OPS[0]));
static const GOp SWIGLU = GLU_OPS[0];
static const GOp GEGLU  = GLU_OPS[3];

static void emit(const GOp & u, const ggml_tensor & s0, const ggml_tensor * s1, const ggml_tensor & d,
                 bool direct, uint32_t th, size_t vt) {
    ggml_hexagon_session sess;
    sess.n_threads = th;
    sess.vtcm_size = vt;

    ggml_tensor op = d;
    op.op     = GGML_OP_GLU;
    op.src[0] = const_cast<ggml_tensor *>(&s0);
    op.src[1] = const_cast<ggml_tensor *>(s1);
    if (u.glu >= 0) {
        op.op_params[0] = u.glu;
        if (htp_op_remap(&op) != u.op) {
            fprintf(stderr, "oracle bug: op remap of %s\n", eltwise_op_name(u.op));
            abort();
        }
    }

    oracle_print_tensor(s0); printf(" | "); print_tensor_or_dash(s1); printf(" | "); oracle_print_tensor(d);
    printf(" | %s %u %s %u %zu => ", eltwise_op_name(u.op), (unsigned) u.op, direct ? "direct" : "node", th, vt);

    if (direct) {
        htp_glu_kernel_params k;
        memset(&k, 0xAA, sizeof(k));
        const bool ok = htp_kparams_solve_unary_mul_glu(&sess, u.op, &s0, s1, &d, &k);
        oracle_print_tensor(s0); printf(" | "); print_tensor_or_dash(s1); printf(" | "); oracle_print_tensor(d); printf(" |");
        oracle_print_words(ok, &k, sizeof(k));
    } else {
        htp_opnode node(u.op, &op);
        const bool ok = htp_kparams_solve(&sess, node);
        if (u.glu >= 0) {
            // the public entry copies a whole kernel_params blob (32 words) for a node-path op
            int32_t k2[HTP_OP_MAX_KERN_PARAMS] = {};
            const bool ok2 = htp_kparams_solve(&sess, &op, k2);
            if (ok != ok2 || (ok && memcmp(k2, node.kernel_params, sizeof(k2)) != 0)) {
                fprintf(stderr, "oracle bug: node path and htp_kparams_solve disagree for %s\n", eltwise_op_name(u.op));
                abort();
            }
        }
        oracle_print_tensor(*node.inputs[0]); printf(" | ");
        print_tensor_or_dash(node.inputs.size() > 1 ? node.inputs[1] : nullptr); printf(" | ");
        oracle_print_tensor(*node.outputs[0]); printf(" |");
        oracle_print_words(ok, node.kernel_params, sizeof(node.kernel_params));
    }
    n_cases++;
}

// dual-source: src0, src1 and dst all [w, r, p, b]
static void emit_dual(const GOp & u, ggml_type t, int64_t w, int64_t r, int64_t p, int64_t b, int k0, int k1, int kd,
                      bool direct, uint32_t th, size_t vt) {
    ggml_tensor s0 = mk(t, w, r, p, b, k0), s1 = mk(t, w, r, p, b, k1), d = mk(t, w, r, p, b, kd);
    emit(u, s0, &s1, d, direct, th, vt);
}

// single-source: src0 [2w, r, p, b] holds gate and linear halves, dst [w, r, p, b]
static void emit_single(const GOp & u, ggml_type t, int64_t w, int64_t r, int64_t p, int64_t b, int k0, int kd,
                        bool direct, uint32_t th, size_t vt) {
    ggml_tensor s0 = mk(t, 2 * w, r, p, b, k0), d = mk(t, w, r, p, b, kd);
    emit(u, s0, nullptr, d, direct, th, vt);
}

struct Run { uint32_t th; size_t vt; };
typedef std::array<int64_t, 4> S4;

static int rot = 0;

int main() {
    const ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16 };
    const int64_t widths[]  = { 1, 3, 4, 8, 16, 31, 32, 64, 100, 128, 576, 1024, 2048, 4096, 8192, 32768 };
    const int64_t shapes[][3] = { { 1, 1, 1 }, { 2, 1, 1 }, { 3, 1, 1 }, { 7, 1, 1 }, { 17, 1, 1 }, { 64, 1, 1 },
                                  { 512, 1, 1 }, { 1000, 1, 1 }, { 4, 3, 1 }, { 5, 2, 2 }, { 9, 3, 2 }, { 2, 2, 2 } };
    const Run      lite[] = { { 6, 8u << 20 }, { 1, 8u << 20 }, { 4, 64u << 10 } };
    const uint32_t ths[]  = { 1, 2, 4, 6, 8 };
    const size_t   vts[]  = { 8u << 20, 1u << 20, 64u << 10, 0 };

    // 1. the shape grid with a rotating op, dual-source
    for (ggml_type t : types) for (int64_t w : widths) for (auto & s : shapes) for (const Run & ru : lite)
        emit_dual(GLU_OPS[rot++ % N_OPS], t, w, s[0], s[1], s[2], 0, 0, 0, false, ru.th, ru.vt);

    // 2. every op, both forms, both types, on a spread of shapes
    for (const GOp & u : GLU_OPS) for (ggml_type t : types)
        for (const S4 & sh : std::vector<S4>{ { 8192, 1, 1, 1 }, { 576, 7, 1, 1 }, { 16, 100, 1, 1 }, { 100, 3, 2, 2 }, { 4096, 3, 1, 1 } })
            for (const Run & ru : lite) {
                emit_dual(u, t, sh[0], sh[1], sh[2], sh[3], 0, 0, 0, false, ru.th, ru.vt);
                emit_single(u, t, sh[0], sh[1], sh[2], sh[3], 0, 0, false, ru.th, ru.vt);
            }

    // 3. the full thread-count by budget grid
    for (const GOp & u : { SWIGLU, GEGLU })
        for (const S4 & sh : std::vector<S4>{ { 8192, 1, 1, 1 }, { 100, 17, 1, 1 }, { 16, 1000, 1, 1 }, { 4096, 3, 1, 1 }, { 32768, 1, 1, 1 } })
            for (uint32_t th : ths) for (size_t vt : vts) {
                emit_dual(u, GGML_TYPE_F32, sh[0], sh[1], sh[2], sh[3], 0, 0, 0, false, th, vt);
                emit_single(u, GGML_TYPE_F32, sh[0], sh[1], sh[2], sh[3], 0, 0, false, th, vt);
            }

    // 4. the collapse steps, as for the unary ops, at every thread count
    {
        const int64_t cs[][4] = { { 8, 64, 1, 1 }, { 4, 3, 5, 2 }, { 1, 7, 3, 2 }, { 16, 1, 5, 3 }, { 1, 1, 1, 1 },
                                  { 31, 66, 1, 1 }, { 16, 1000, 1, 1 }, { 30, 77, 2, 1 }, { 30, 64, 3, 1 }, { 3, 1024, 1, 1 },
                                  { 64, 33, 1, 1 }, { 64, 32, 1, 1 }, { 64, 31, 1, 1 }, { 32, 65, 1, 1 }, { 127, 10, 1, 1 },
                                  { 8192, 1, 1, 1 }, { 4096, 2, 1, 1 }, { 2048, 5, 1, 1 }, { 16384, 3, 1, 1 }, { 1024, 1, 1, 1 },
                                  { 1023, 1, 1, 1 }, { 4100, 1, 1, 1 }, { 4096, 1, 1, 1 }, { 12288, 1, 1, 1 }, { 6000, 1, 1, 1 },
                                  { 8192, 1, 2, 1 }, { 8192, 2, 2, 1 }, { 5000, 1, 3, 1 }, { 2048, 1, 1, 7 }, { 20000, 1, 1, 1 },
                                  { 192, 3, 1, 1 }, { 96, 5, 1, 1 }, { 4, 512, 1, 1 }, { 4, 2048, 1, 1 }, { 33, 63, 1, 1 },
                                  { 24, 85, 1, 1 }, { 2, 1024, 2, 2 }, { 2, 4, 1024, 1 } };
        for (ggml_type t : types) for (const GOp & u : { SWIGLU, GEGLU }) for (auto & c : cs) for (uint32_t th : ths) {
            emit_dual(u, t, c[0], c[1], c[2], c[3], 0, 0, 0, false, th, 8u << 20);
            emit_dual(u, t, c[0], c[1], c[2], c[3], 0, 0, 0, true, th, 8u << 20);
        }
        for (int kind : { 1, 3, 4, 5 }) for (auto & c : cs) for (uint32_t th : { 1, 6 })
            emit_dual(SWIGLU, GGML_TYPE_F32, c[0], c[1], c[2], c[3], kind, kind, 0, false, th, 8u << 20);
        for (auto & c : cs) for (size_t vt : { 1u << 20, 64u << 10, 16u << 10 }) for (uint32_t th : { 1, 8 })
            emit_dual(SWIGLU, GGML_TYPE_F32, c[0], c[1], c[2], c[3], 0, 0, 0, false, th, vt);
    }

    // 5. layouts on every operand, both forms
    for (const S4 & sh : std::vector<S4>{ { 96, 5, 1, 1 }, { 96, 4, 3, 1 }, { 100, 3, 2, 2 }, { 8, 64, 1, 1 } }) {
        for (int k0 : { 0, 1, 2, 3, 4, 5 }) for (int k1 : { 0, 1, 2, 3, 4, 5 }) for (int kd : { 0, 1, 3 }) {
            emit_dual(SWIGLU, GGML_TYPE_F32, sh[0], sh[1], sh[2], sh[3], k0, k1, kd, false, 6, 8u << 20);
        }
        for (int k0 : { 0, 1, 2, 3, 4, 5 }) for (int kd : { 0, 1, 2, 3 }) for (ggml_type t : types) {
            emit_single(SWIGLU, t, sh[0], sh[1], sh[2], sh[3], k0, kd, false, 6, 8u << 20);
        }
    }
    for (int k0 : { 0, 1, 4 }) for (int k1 : { 0, 1, 4 })
        emit_dual(GEGLU, GGML_TYPE_F16, 96, 4, 3, 1, k0, k1, 0, false, 6, 8u << 20);

    // 6. mismatched or unsupported operands
    {
        ggml_tensor f32 = mk(GGML_TYPE_F32, 64, 4, 1, 1), f16 = mk(GGML_TYPE_F16, 64, 4, 1, 1);
        ggml_tensor i32 = mk(GGML_TYPE_I32, 64, 4, 1, 1);
        ggml_tensor big = mk(GGML_TYPE_F32, 64, 8, 1, 1), wide = mk(GGML_TYPE_F32, 128, 4, 1, 1);
        ggml_tensor f32p = mk(GGML_TYPE_F32, 64, 4, 1, 1, 1);
        for (const GOp & u : { SWIGLU, GEGLU, GLU_OPS[4], GLU_OPS[7] }) {
            emit(u, f32, &f16, f32, false, 6, 8u << 20);   // src1 type differs
            emit(u, f32, &f32, f16, false, 6, 8u << 20);   // dst type differs
            emit(u, f16, &f16, f32, false, 6, 8u << 20);
            emit(u, i32, &i32, i32, false, 6, 8u << 20);
            emit(u, f32, &big, f32, false, 6, 8u << 20);   // src1 shape differs
            emit(u, f32, &f32, big, false, 6, 8u << 20);   // dst shape differs
            emit(u, f32, &f32, wide, false, 6, 8u << 20);
            emit(u, f32, &f32, f32p, false, 6, 8u << 20);  // dst not contiguous
            emit(u, f32, nullptr, f32, false, 6, 8u << 20); // single-source with a dst as wide as src0
            emit(u, wide, nullptr, f32, false, 6, 8u << 20);
            emit(u, wide, nullptr, f32p, false, 6, 8u << 20);
        }
    }

    // 7. direct calls (no collapse) on the cera shapes and the unsupported opcodes
    for (const GOp & u : GLU_OPS) for (ggml_type t : types)
        for (const S4 & sh : std::vector<S4>{ { 2048, 1, 1, 1 }, { 2048, 512, 1, 1 }, { 64, 32, 1, 1 }, { 8192, 4, 1, 1 }, { 100, 3, 2, 2 } })
            for (const Run & ru : std::vector<Run>{ { 6, 8u << 20 }, { 8, 1u << 20 }, { 4, 64u << 10 }, { 6, 0 } }) {
                emit_dual(u, t, sh[0], sh[1], sh[2], sh[3], 0, 0, 0, true, ru.th, ru.vt);
                emit_single(u, t, sh[0], sh[1], sh[2], sh[3], 0, 0, true, ru.th, ru.vt);
            }
    {
        ggml_hexagon_session sess;
        ggml_tensor s = mk(GGML_TYPE_F32, 64, 4, 1, 1), d = s;
        htp_glu_kernel_params k;
        // an opcode outside the GLU family is rejected whatever the operands are
        if (htp_kparams_solve_unary_mul_glu(&sess, HTP_OP_ADD, &s, &s, &d, &k)) {
            fprintf(stderr, "oracle bug: a non-GLU opcode was accepted\n");
            abort();
        }
    }

    fprintf(stderr, "%d cases\n", n_cases);
    return 0;
}
