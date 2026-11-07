// Case table for ADD / MUL / SUB / DIV / ADD_ID through the real `htp_kparams_solve`. Line format:
//   <src0> | <src1> | <src2 or -> | <dst> | <op name> <opcode> <session threads> <vtcm bytes> => <words | unsupported>
// Binary ops have no collapse step in the host (`htp_kparams_solve` hands them straight to the
// solver), so what is printed is the solver's answer for the tensors as given.
#include "eltwise.h"

#include <vector>

static int n_cases = 0;

static void emit(uint32_t htp_op, ggml_op gop, const ggml_tensor & s0, const ggml_tensor & s1,
                 const ggml_tensor * s2, const ggml_tensor & d, uint32_t th, size_t vt) {
    ggml_hexagon_session sess;
    sess.n_threads = th;
    sess.vtcm_size = vt;

    ggml_tensor op = d;
    op.op     = gop;
    op.src[0] = const_cast<ggml_tensor *>(&s0);
    op.src[1] = const_cast<ggml_tensor *>(&s1);
    op.src[2] = const_cast<ggml_tensor *>(s2);
    if (htp_op_remap(&op) != (htp_op_code) htp_op) {
        fprintf(stderr, "oracle bug: op remap\n");
        abort();
    }

    htp_binary_kernel_params k;
    memset(&k, 0xAA, sizeof(k));  // poison, so a field the solver forgets to set shows up
    const bool ok = htp_kparams_solve(&sess, &op, &k);

    oracle_print_tensor(s0); printf(" | "); oracle_print_tensor(s1); printf(" | ");
    print_tensor_or_dash(s2); printf(" | "); oracle_print_tensor(d);
    printf(" | %s %u %u %zu =>", eltwise_op_name(htp_op), htp_op, th, vt);
    oracle_print_words(ok, &k, sizeof(k));
    n_cases++;
}

struct Run { uint32_t th; size_t vt; };

enum Pat { SAME, ROW, SCALAR, SCALROW, SCALALL, SCALPLANE, PLANEBC, BATCHBC, HALFROWS, REP2, REPROW, N_PAT };

// src1 for a pattern, relative to src0 = [w, r, p, b]; false when the pattern degenerates for this shape
static bool src1_for(Pat pat, ggml_type t, int64_t w, int64_t r, int64_t p, int64_t b, int kind, ggml_tensor * out) {
    switch (pat) {
        case SAME:      *out = mk(t, w, r, p, b, kind); return true;
        case ROW:       *out = mk(t, w, 1, 1, 1, kind); return true;
        case SCALAR:    *out = mk(t, 1, 1, 1, 1, kind); return true;
        case SCALROW:   *out = mk(t, 1, r, 1, 1, kind); return true;
        case SCALALL:   *out = mk(t, 1, r, p, b, kind); return true;
        case SCALPLANE: *out = mk(t, 1, 1, p, 1, kind); return true;
        case PLANEBC:   *out = mk(t, w, 1, p, 1, kind); return true;
        case BATCHBC:   *out = mk(t, w, r, 1, b, kind); return true;
        case HALFROWS:
            if (r % 2 != 0 || r < 4) return false;
            *out = mk(t, w, 2, p, b, kind);
            return true;
        case REP2:
            if (w % 2 != 0) return false;
            *out = mk(t, w / 2, r, p, b, kind);
            return true;
        case REPROW:
            if (w % 2 != 0) return false;
            *out = mk(t, w / 2, 1, 1, 1, kind);
            return true;
        default: return false;
    }
}

static const uint32_t OPS[4]     = { HTP_OP_ADD, HTP_OP_MUL, HTP_OP_SUB, HTP_OP_DIV };
static const ggml_op  GOPS[4]    = { GGML_OP_ADD, GGML_OP_MUL, GGML_OP_SUB, GGML_OP_DIV };
static int            op_rot     = 0;

static void emit_bin(ggml_type t, int64_t w, int64_t r, int64_t p, int64_t b, Pat pat, int k0, int k1, int kd,
                     uint32_t th, size_t vt) {
    ggml_tensor s1;
    if (!src1_for(pat, t, w, r, p, b, k1, &s1)) return;
    ggml_tensor s0 = mk(t, w, r, p, b, k0);
    ggml_tensor d  = mk(t, w, r, p, b, kd);
    const int   i  = op_rot++ % 4;
    emit(OPS[i], GOPS[i], s0, s1, nullptr, d, th, vt);
}

int main() {
    const ggml_type types[]  = { GGML_TYPE_F32, GGML_TYPE_F16 };
    const int64_t   widths[] = { 1, 7, 32, 33, 100, 128, 576, 2048, 8192, 65536 };
    const int64_t   shapes[][3] = { { 1, 1, 1 }, { 3, 1, 1 }, { 17, 1, 1 }, { 64, 1, 1 }, { 512, 1, 1 },
                                    { 4, 3, 1 }, { 5, 2, 2 }, { 9, 3, 2 } };
    const Run lite[]  = { { 6, 8u << 20 }, { 4, 64u << 10 } };
    const uint32_t ths[] = { 1, 2, 4, 6, 8 };
    const size_t   vts[] = { 8u << 20, 1u << 20, 64u << 10, 0 };

    // 1. contiguous tensors: every shape, every broadcast pattern, both types
    for (ggml_type t : types) for (int64_t w : widths) for (auto & s : shapes) for (int pat = 0; pat < N_PAT; pat++)
        for (const Run & ru : lite)
            emit_bin(t, w, s[0], s[1], s[2], (Pat) pat, 0, 0, 0, ru.th, ru.vt);

    // 2. the full thread-count by VTCM-budget grid on a spread of shapes
    for (int64_t w : std::vector<int64_t>{ 100, 2048 }) for (int si : { 0, 2, 5, 7 }) for (int pat = 0; pat < N_PAT; pat++)
        for (uint32_t th : ths) for (size_t vt : vts)
            emit_bin(GGML_TYPE_F32, w, shapes[si][0], shapes[si][1], shapes[si][2], (Pat) pat, 0, 0, 0, th, vt);

    // 3. padded, strided, transposed and permuted layouts on every operand
    for (int64_t w : std::vector<int64_t>{ 96, 1024 }) for (int si : { 2, 5, 6 }) for (int pat : { SAME, ROW, SCALAR, SCALROW, PLANEBC, REP2 })
        for (int k0 : { 0, 1, 2, 3, 4, 5 }) for (int k1 : { 0, 1, 2, 4 }) for (int kd : { 0, 1, 3 })
            emit_bin(GGML_TYPE_F32, w, shapes[si][0], shapes[si][1], shapes[si][2], (Pat) pat, k0, k1, kd, 6, 8u << 20);
    for (int pat : { SAME, ROW, SCALROW, PLANEBC }) for (int k0 : { 0, 1, 4 }) for (int k1 : { 0, 1 }) for (int kd : { 0, 1 })
        emit_bin(GGML_TYPE_F16, 96, 4, 3, 1, (Pat) pat, k0, k1, kd, 6, 8u << 20);

    // 4. every op on the same few shapes
    for (int i = 0; i < 4; i++) for (ggml_type t : types) for (int pat : { SAME, ROW, SCALAR, SCALROW, REP2 }) {
        ggml_tensor s1;
        if (!src1_for((Pat) pat, t, 256, 8, 3, 1, 0, &s1)) continue;
        ggml_tensor s0 = mk(t, 256, 8, 3, 1), d = s0;
        emit(OPS[i], GOPS[i], s0, s1, nullptr, d, 6, 8u << 20);
    }

    // 5. mixed types, mismatched dst shapes, a non-repeatable src1, other element types
    {
        ggml_tensor f32 = mk(GGML_TYPE_F32, 64, 4, 1, 1), f16 = mk(GGML_TYPE_F16, 64, 4, 1, 1);
        ggml_tensor i32 = mk(GGML_TYPE_I32, 64, 4, 1, 1), q8 = mk(GGML_TYPE_Q8_0, 64, 4, 1, 1);
        ggml_tensor big = mk(GGML_TYPE_F32, 64, 8, 1, 1), odd = mk(GGML_TYPE_F32, 48, 4, 1, 1);
        ggml_tensor oddr = mk(GGML_TYPE_F32, 64, 3, 1, 1), z = mk(GGML_TYPE_F32, 64, 4, 1, 1);
        emit(HTP_OP_ADD, GGML_OP_ADD, f32, f16, nullptr, f32, 6, 8u << 20);
        emit(HTP_OP_ADD, GGML_OP_ADD, f32, f32, nullptr, f16, 6, 8u << 20);
        emit(HTP_OP_ADD, GGML_OP_ADD, f16, f32, nullptr, f16, 6, 8u << 20);
        emit(HTP_OP_ADD, GGML_OP_ADD, i32, i32, nullptr, i32, 6, 8u << 20);
        emit(HTP_OP_MUL, GGML_OP_MUL, q8, q8, nullptr, q8, 6, 8u << 20);
        emit(HTP_OP_MUL, GGML_OP_MUL, f32, f32, nullptr, big, 6, 8u << 20);
        emit(HTP_OP_MUL, GGML_OP_MUL, f32, odd, nullptr, f32, 6, 8u << 20);   // 64 % 48 != 0
        emit(HTP_OP_MUL, GGML_OP_MUL, f32, oddr, nullptr, f32, 6, 8u << 20);  // 4 % 3 != 0
        emit(HTP_OP_SUB, GGML_OP_SUB, big, f32, nullptr, big, 6, 8u << 20);   // src1 broadcast over rows, ne1 % ne1' == 0
        emit(HTP_OP_DIV, GGML_OP_DIV, f32, big, nullptr, f32, 6, 8u << 20);   // src1 larger than src0
        (void) z;
    }

    // 6. empty and single-element tensors that stay on the flat chunked kernel
    for (uint32_t th : { 1, 6 }) {
        ggml_tensor e0 = mk(GGML_TYPE_F32, 0, 4, 1, 1), e1 = mk(GGML_TYPE_F32, 8, 0, 1, 1);
        emit(HTP_OP_ADD, GGML_OP_ADD, e0, e0, nullptr, e0, th, 8u << 20);
        emit(HTP_OP_ADD, GGML_OP_ADD, e1, e1, nullptr, e1, th, 8u << 20);
        ggml_tensor one = mk(GGML_TYPE_F32, 1, 1, 1, 1);
        emit(HTP_OP_MUL, GGML_OP_MUL, one, one, nullptr, one, th, 8u << 20);
    }

    // 7. ADD_ID: src0 [w, n_expert_used, n_tokens], src1 [w, n_experts], src2 I32 [n_expert_used, n_tokens]
    for (int64_t w : std::vector<int64_t>{ 64, 256, 2048 }) for (int64_t ne : std::vector<int64_t>{ 2, 4, 8 }) for (int64_t nt : std::vector<int64_t>{ 1, 5, 64 })
        for (int64_t rows : std::vector<int64_t>{ 8, 32, 128, 256 }) for (const Run & ru : std::vector<Run>{ { 6, 8u << 20 }, { 1, 8u << 20 }, { 8, 64u << 10 }, { 6, 0 } }) {
            ggml_tensor s0 = mk(GGML_TYPE_F32, w, ne, nt, 1), s1 = mk(GGML_TYPE_F32, w, rows, 1, 1);
            ggml_tensor s2 = mk(GGML_TYPE_I32, ne, nt, 1, 1), d = s0;
            emit(HTP_OP_ADD_ID, GGML_OP_ADD_ID, s0, s1, &s2, d, ru.th, ru.vt);
        }
    {
        const int64_t w = 128, ne = 4, nt = 8, rows = 16;
        ggml_tensor s0 = mk(GGML_TYPE_F32, w, ne, nt, 1), s1 = mk(GGML_TYPE_F32, w, rows, 1, 1);
        ggml_tensor s2 = mk(GGML_TYPE_I32, ne, nt, 1, 1), d = s0;
        ggml_tensor s0p = mk(GGML_TYPE_F32, w, ne, nt, 1, 1), s1p = mk(GGML_TYPE_F32, w, rows, 1, 1, 1);
        ggml_tensor s2p = mk(GGML_TYPE_I32, ne, nt, 1, 1, 1), dp = mk(GGML_TYPE_F32, w, ne, nt, 1, 1);
        ggml_tensor s0h = mk(GGML_TYPE_F16, w, ne, nt, 1), s1h = mk(GGML_TYPE_F16, w, rows, 1, 1), dh = s0h;
        ggml_tensor s1w = mk(GGML_TYPE_F32, w / 2, rows, 1, 1), s2w = mk(GGML_TYPE_I32, ne + 1, nt, 1, 1);
        ggml_tensor s2t = mk(GGML_TYPE_I32, ne, nt + 1, 1, 1), s1b = mk(GGML_TYPE_F32, w, rows, 2, 1);
        ggml_tensor s2f = mk(GGML_TYPE_F32, ne, nt, 1, 1), dbig = mk(GGML_TYPE_F32, w, ne, nt + 1, 1);
        emit(HTP_OP_ADD_ID, GGML_OP_ADD_ID, s0, s1, &s2, d, 6, 8u << 20);
        emit(HTP_OP_ADD_ID, GGML_OP_ADD_ID, s0p, s1, &s2, d, 6, 8u << 20);   // src0 row-padded
        emit(HTP_OP_ADD_ID, GGML_OP_ADD_ID, s0, s1p, &s2, d, 6, 8u << 20);   // src1 row-padded
        emit(HTP_OP_ADD_ID, GGML_OP_ADD_ID, s0, s1, &s2p, d, 6, 8u << 20);   // src2 row-padded (allowed)
        emit(HTP_OP_ADD_ID, GGML_OP_ADD_ID, s0, s1, &s2, dp, 6, 8u << 20);   // dst row-padded
        emit(HTP_OP_ADD_ID, GGML_OP_ADD_ID, s0h, s1h, &s2, dh, 6, 8u << 20); // F16
        emit(HTP_OP_ADD_ID, GGML_OP_ADD_ID, s0, s1w, &s2, d, 6, 8u << 20);   // src1 width mismatch
        emit(HTP_OP_ADD_ID, GGML_OP_ADD_ID, s0, s1, &s2w, d, 6, 8u << 20);   // ids per token mismatch
        emit(HTP_OP_ADD_ID, GGML_OP_ADD_ID, s0, s1, &s2t, d, 6, 8u << 20);   // token count mismatch
        emit(HTP_OP_ADD_ID, GGML_OP_ADD_ID, s0, s1b, &s2, d, 6, 8u << 20);   // src1 has a plane dimension
        emit(HTP_OP_ADD_ID, GGML_OP_ADD_ID, s0, s1, &s2f, d, 6, 8u << 20);   // ids not I32
        emit(HTP_OP_ADD_ID, GGML_OP_ADD_ID, s0, s1, &s2, dbig, 6, 8u << 20); // dst shape differs
        emit(HTP_OP_ADD_ID, GGML_OP_ADD_ID, s0, s1, nullptr, d, 6, 8u << 20); // no ids tensor
    }

    fprintf(stderr, "%d cases\n", n_cases);
    return 0;
}
