// Case table for SUM, SUM_ROWS and ARGMAX through the real `htp_kparams_solve` (they share one
// solver). Line format:
//   <op: 0 SUM, 1 SUM_ROWS, 2 ARGMAX> <src0> | <dst> | <session threads> => <2 words | unsupported>
// The words are `htp_sum_rows_kernel_params`: n_threads, elems_per_thread.
//
// Session thread count 0 is only emitted where the solver does not divide by it (a zero-thread
// session divides by zero for a SUM of 1024 or more elements and an ARGMAX over one long row).
#include "common.h"

namespace {

int n_cases = 0;

enum { OP_SUM = 0, OP_SUM_ROWS = 1, OP_ARGMAX = 2 };

void emit(int which, const ggml_tensor & s, const ggml_tensor & d, uint32_t th) {
    const int64_t n = s.ne[0] * s.ne[1] * s.ne[2] * s.ne[3];
    if (th == 0 && ((which == OP_SUM && n >= 1024) || (which == OP_ARGMAX && s.ne[1] * s.ne[2] * s.ne[3] <= 1 && s.ne[0] >= 1024))) {
        return;
    }
    ggml_hexagon_session sess;
    sess.n_threads = th;

    ggml_tensor op = d;
    op.op     = which == OP_SUM ? GGML_OP_SUM : which == OP_SUM_ROWS ? GGML_OP_SUM_ROWS : GGML_OP_ARGMAX;
    op.src[0] = const_cast<ggml_tensor *>(&s);

    htp_sum_rows_kernel_params k;
    memset(&k, 0xAA, sizeof(k));
    const bool ok = htp_kparams_solve(&sess, &op, &k);

    printf("%d ", which);
    oracle_print_tensor(s); printf(" | "); oracle_print_tensor(d);
    printf(" | %u =>", th);
    oracle_print_words(ok, &k, sizeof(k), sizeof(k) / 4);
    n_cases++;
}

// SUM reduces to one element, SUM_ROWS to [1, ne1, ne2, ne3], ARGMAX to one I32 per row.
void emit_shape(int which, ggml_type st, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, int kind, uint32_t th) {
    const int64_t s_ne[4] = { ne0, ne1, ne2, ne3 };
    int64_t d_ne[4];
    ggml_type dt = GGML_TYPE_F32;
    switch (which) {
        case OP_SUM:      d_ne[0] = 1; d_ne[1] = 1; d_ne[2] = 1; d_ne[3] = 1; break;
        case OP_SUM_ROWS: d_ne[0] = 1; d_ne[1] = ne1; d_ne[2] = ne2; d_ne[3] = ne3; break;
        default:          d_ne[0] = ne1; d_ne[1] = ne2; d_ne[2] = ne3; d_ne[3] = 1; dt = GGML_TYPE_I32; break;
    }
    ggml_tensor s = oracle_tensor(st, s_ne, kind);
    ggml_tensor d = oracle_tensor(dt, d_ne);
    emit(which, s, d, th);
}

}  // namespace

int main() {
    const uint32_t threads[] = { 0, 1, 4, 6, 8, 10, 12 };

    for (int which : { OP_SUM, OP_SUM_ROWS, OP_ARGMAX }) {
        // One row, as greedy sampling over a vocabulary: every width around the 1024 threshold and
        // the 32-element rounding.
        for (int64_t ne0 : { 1, 2, 31, 32, 33, 100, 1000, 1023, 1024, 1025, 2047, 2048, 4096, 32000, 65536, 65537, 128256,
                             151936, 262144 })
        for (uint32_t th : threads) {
            emit_shape(which, GGML_TYPE_F32, ne0, 1, 1, 1, 0, th);
        }
        // Many rows: a batch of logits rows, and the row-count thresholds of the general path.
        for (int64_t ne0 : { 1, 32, 100, 255, 256, 257, 1024, 4096 })
        for (int64_t ne1 : { 1, 2, 3, 31, 32, 33, 63, 64, 65, 100, 513 })
        for (int64_t ne2 : { 1, 3 }) for (int64_t ne3 : { 1, 2 }) for (int kind : { 0, 1 })
        for (uint32_t th : { 0u, 1u, 4u, 8u, 12u }) {
            if (ne0 * ne1 * ne2 * ne3 > 20 * 1000 * 1000) continue;
            emit_shape(which, GGML_TYPE_F32, ne0, ne1, ne2, ne3, kind, th);
        }
        // Row strides either side of the 1024-byte threshold (nb[1] < 1024 picks 32 rows per thread).
        for (int64_t ne0 : { 250, 255, 256, 257, 260 }) for (int64_t ne1 : { 8, 64, 640 }) for (uint32_t th : { 4u, 8u }) {
            emit_shape(which, GGML_TYPE_F32, ne0, ne1, 1, 1, 0, th);
        }
    }

    // Refusals: a non-F32 source, a destination of the wrong type, and non-contiguous operands.
    for (int which : { OP_SUM, OP_SUM_ROWS, OP_ARGMAX }) {
        const int64_t s_ne[4] = { 256, 4, 1, 1 };
        const int64_t d_ne[4] = { which == OP_SUM_ROWS ? 1 : 4, which == OP_SUM_ROWS ? 4 : 1, 1, 1 };
        for (ggml_type st : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I32, GGML_TYPE_BF16, GGML_TYPE_Q8_0 }) {
            for (ggml_type dt : { GGML_TYPE_F32, GGML_TYPE_I32, GGML_TYPE_F16, GGML_TYPE_I64 }) {
                ggml_tensor s = oracle_tensor(st, s_ne);
                ggml_tensor d = oracle_tensor(dt, d_ne);
                emit(which, s, d, 6);
            }
        }
        ggml_tensor d = oracle_tensor(which == OP_ARGMAX ? GGML_TYPE_I32 : GGML_TYPE_F32, d_ne);
        ggml_tensor padded = oracle_tensor(GGML_TYPE_F32, s_ne, 1);
        ggml_tensor trans  = oracle_tensor(GGML_TYPE_F32, s_ne, 2);
        ggml_tensor perm   = oracle_tensor(GGML_TYPE_F32, s_ne, 3);
        emit(which, padded, d, 6);
        emit(which, trans, d, 6);
        emit(which, perm, d, 6);
        // a destination whose elements are strided
        const size_t dnb[4] = { 8, 32, 32, 32 };
        ggml_tensor dstr = oracle_tensor_nb(which == OP_ARGMAX ? GGML_TYPE_I32 : GGML_TYPE_F32, d_ne, dnb);
        ggml_tensor s = oracle_tensor(GGML_TYPE_F32, s_ne);
        emit(which, s, dstr, 6);
        ggml_tensor dpad = oracle_tensor(which == OP_ARGMAX ? GGML_TYPE_I32 : GGML_TYPE_F32, d_ne, 1);
        emit(which, s, dpad, 6);
    }

    fprintf(stderr, "%d cases\n", n_cases);
    return 0;
}
