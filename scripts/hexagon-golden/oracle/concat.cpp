// Case table for CONCAT through the real `htp_kparams_solve`. Line format:
//   <src0> | <src1> | <dst> | <dim> <session threads> <session vtcm bytes> => <words | unsupported>
// The words are those of `htp_concat_kernel_params`, trailing zeros omitted.
#include "common.h"

namespace {

int n_cases = 0;

// A dense tensor with extents `ne`; `kind` as in `oracle_tensor`, plus 4: rows `ne[0]` apart by
// two spare elements and every outer stride padded too (a view of a wider tensor).
ggml_tensor make(ggml_type t, const int64_t * ne, int kind) {
    if (kind < 4) {
        return oracle_tensor(t, ne, kind);
    }
    const size_t ts = ggml_type_size(t);
    const size_t nb[4] = { ts, (ne[0] + 2) * ts, (ne[0] + 2) * ts * ne[1] + 64, ((ne[0] + 2) * ts * ne[1] + 64) * ne[2] };
    return oracle_tensor_nb(t, ne, nb);
}

void print_trimmed(bool ok, const void * kparams, size_t bytes) {
    if (!ok) {
        printf(" unsupported\n");
        return;
    }
    uint32_t w[32] = {};
    memcpy(w, kparams, bytes < sizeof(w) ? bytes : sizeof(w));
    int n = bytes / 4;
    while (n > 1 && w[n - 1] == 0) n--;
    for (int i = 0; i < n; i++) printf(" %u", w[i]);
    printf("\n");
}

void emit(const ggml_tensor & a, const ggml_tensor & b, const ggml_tensor & d, int dim, uint32_t th, size_t vt) {
    // A session without threads divides by zero in the gather-rows planner (the real host never
    // has one), so only the shapes that cannot take that kernel run without threads.
    const size_t ts = ggml_type_size(d.type);
    if (th == 0 && dim == 0 && d.ne[0] <= (int64_t) (128 / ts)) {
        return;
    }
    ggml_hexagon_session sess;
    sess.n_threads = th;
    sess.vtcm_size = vt;

    ggml_tensor op = d;
    op.op     = GGML_OP_CONCAT;
    op.src[0] = const_cast<ggml_tensor *>(&a);
    op.src[1] = const_cast<ggml_tensor *>(&b);
    op.op_params[0] = dim;
    htp_concat_kernel_params k;
    memset(&k, 0xAA, sizeof(k));  // poison, so a field the solver forgets to set shows up
    const bool ok = htp_kparams_solve(&sess, &op, &k);

    oracle_print_tensor(a); printf(" | "); oracle_print_tensor(b); printf(" | "); oracle_print_tensor(d);
    printf(" | %d %u %zu =>", dim, th, vt);
    print_trimmed(ok, &k, sizeof(k));
    n_cases++;
}

// a, b, d extents for a join along `dd` (b longer by 3 along it, so the sources differ)
void joined(const int64_t * base, int dd, int64_t * ea, int64_t * eb, int64_t * ed) {
    for (int i = 0; i < 4; i++) { ea[i] = eb[i] = ed[i] = base[i]; }
    eb[dd] = base[dd] + 3;
    ed[dd] = ea[dd] + eb[dd];
}

}  // namespace

int main() {
    const ggml_type types[]  = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I32 };
    const uint32_t threads[] = { 0, 1, 4, 8 };
    const size_t   vtcms[]   = { 8u << 20, 32u << 10, 0 };

    // Every dim (and the out-of-range -1 and 4), every operand layout: dense, row-padded,
    // transposed, dim 2 inside dim 1, and wider-view strides. The full thread and budget table
    // runs for the layouts that pick a threaded kernel, one session for the rest.
    const int64_t bases[][4] = {
        { 6, 4, 1, 1 }, { 64, 3, 2, 1 }, { 1, 1, 1, 1 }, { 33, 7, 1, 1 }, { 256, 16, 1, 1 },
        { 2, 100, 1, 1 }, { 5, 9, 2, 3 }, { 13, 2, 3, 1 },
    };
    for (ggml_type t : types) for (const auto & base : bases) for (int dim = -1; dim <= 4; dim++) {
        const int dd = dim < 0 || dim > 3 ? 0 : dim;
        const bool f32 = t == GGML_TYPE_F32;
        if (!f32 && (&base - bases) % 2 == 0) continue;  // the other dtypes on half of the shapes
        int64_t ea[4], eb[4], ed[4];
        joined(base, dd, ea, eb, ed);
        for (int ka = 0; ka < 5; ka++) for (int kb = 0; kb < 5; kb++) for (int kd = 0; kd < 4; kd++) {
            if (!f32 && (ka > 1 || kd > 1)) continue;
            ggml_tensor a = make(t, ea, ka), b = make(t, eb, kb), d = make(t, ed, kd);
            const bool threaded = f32 && dd == 0 && kb == 2 && ka == 0 && kd == 0;
            for (uint32_t th : threads) for (size_t vt : vtcms) {
                if (!threaded && (th != 4 || vt != (8u << 20))) continue;
                emit(a, b, d, dim, th, vt);
            }
        }
    }

    // Short rows joined along dim 0 (a conv state ahead of new positions): the gather-rows
    // kernel, over row widths on both sides of its limit, row counts, batch dims, both ways of
    // reading the second operand, and the budgets that shrink or refuse its run of rows.
    const int64_t na_[] = { 1, 2, 3, 7, 16, 31, 32 };
    const int64_t nb_[] = { 1, 3, 8, 31, 33, 64 };
    const int64_t rows_[] = { 0, 1, 3, 32, 100, 4096 };
    const int64_t batch_[][2] = { { 1, 1 }, { 2, 3 } };
    for (ggml_type t : { GGML_TYPE_F32, GGML_TYPE_F16 })
    for (int64_t na : na_) for (int64_t nb : nb_) for (int64_t rows : rows_) for (const auto & bt : batch_) {
        const int64_t ea[4] = { na, rows, bt[0], bt[1] }, eb[4] = { nb, rows, bt[0], bt[1] }, ed[4] = { na + nb, rows, bt[0], bt[1] };
        // a dense or row-padded state, and a dense or transposed second source (the audio
        // encoders' pads read an activation through its own strides)
        for (int kb : { 0, 2 }) for (int ka : { 0, 1 }) {
            const bool all_sessions = (rows == 100 || rows == 4096) && bt[0] == 1 && (nb == 8 || nb == 33) && ka == 0 && kb == 2;
            for (uint32_t th : threads) for (size_t vt : vtcms) {
                if (!all_sessions && (th != 4 || vt != (8u << 20))) continue;
                ggml_tensor a = make(t, ea, ka), b = make(t, eb, kb), d = make(t, ed, 0);
                emit(a, b, d, 0, th, vt);
            }
        }
    }
    // the conformer convolution module: pad + time-major/channel-major sources of `dim` channels
    // (f16 too, and sources longer than the transposed kernel's token block)
    for (ggml_type t : { GGML_TYPE_F32, GGML_TYPE_F16 }) for (int64_t dim : { 16, 176, 1024 }) for (int64_t r0 : { 2, 4, 17, 100 })
    for (int64_t r1 : { 1, 4, 17, 100, 400, 700 }) for (int kb : { 0, 2 }) for (uint32_t th : { 1u, 8u }) for (size_t vt : { size_t(8u << 20), size_t(64u << 10) }) {
        if (t == GGML_TYPE_F16 && dim == 176) continue;
        const int64_t ea[4] = { r0, dim, 1, 1 }, eb[4] = { r1, dim, 1, 1 }, ed[4] = { r0 + r1, dim, 1, 1 };
        ggml_tensor a = oracle_tensor(t, ea, 0), b = oracle_tensor(t, eb, kb), d = oracle_tensor(t, ed, 0);
        emit(a, b, d, 0, th, vt);
    }

    // Join-shaped operands the kernels refuse
    {
        const int64_t ea[4] = { 6, 4, 1, 1 }, eb[4] = { 2, 4, 1, 1 }, ed[4] = { 7, 4, 1, 1 };
        ggml_tensor a = make(GGML_TYPE_F32, ea, 0), b = make(GGML_TYPE_F32, eb, 0), d = make(GGML_TYPE_F32, ed, 0);
        emit(a, b, d, 0, 4, 8u << 20);               // dst one short
        ggml_tensor h = make(GGML_TYPE_F16, eb, 0);
        emit(a, h, d, 0, 4, 8u << 20);               // mixed types
        const int64_t ed2[4] = { 8, 4, 1, 1 };
        ggml_tensor d2 = make(GGML_TYPE_F32, ed2, 0);
        emit(a, b, d2, 1, 4, 8u << 20);              // dim 1 with the wrong extents
        const int64_t eb2[4] = { 2, 5, 1, 1 };
        ggml_tensor b2 = make(GGML_TYPE_F32, eb2, 0);
        emit(a, b2, d2, 0, 4, 8u << 20);             // src1 off along a dim that is not joined
        const ggml_type bad[] = { GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_I64, GGML_TYPE_F64 };
        for (ggml_type t : bad) {
            const int64_t xa[4] = { 32, 4, 1, 1 }, xb[4] = { 32, 4, 1, 1 }, xd[4] = { 64, 4, 1, 1 };
            ggml_tensor ba = oracle_tensor(t, xa, 0), bb = oracle_tensor(t, xb, 0), bd = oracle_tensor(t, xd, 0);
            emit(ba, bb, bd, 0, 4, 8u << 20);
        }
    }

    fprintf(stderr, "%d cases\n", n_cases);
    return 0;
}
