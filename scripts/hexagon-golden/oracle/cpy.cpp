// Case table for CPY through the real `htp_kparams_solve`. Line format:
//   <src> | <dst> | <session threads> <session vtcm bytes> => <words | unsupported>
// The words are those of `htp_copy_kernel_params`, trailing zeros omitted.
#include "common.h"

namespace {

int n_cases = 0;

// A dense tensor with extents `ne`; `kind` as in `oracle_tensor`, plus 4: rows two spare
// elements apart and padded outer strides (a view of a wider tensor).
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

void emit(const ggml_tensor & s, const ggml_tensor & d, uint32_t th, size_t vt) {
    // A session without threads divides by zero in the gather-rows planner (the real host never
    // has one), so only the shapes that cannot take that kernel run without threads.
    if (th == 0 && s.type == d.type && s.ne[0] <= (int64_t) (128 / ggml_type_size(s.type))) {
        return;
    }
    ggml_hexagon_session sess;
    sess.n_threads = th;
    sess.vtcm_size = vt;

    ggml_tensor op = d;
    op.op     = GGML_OP_CPY;
    op.src[0] = const_cast<ggml_tensor *>(&s);
    htp_copy_kernel_params k;
    memset(&k, 0xAA, sizeof(k));  // poison, so a field the solver forgets to set shows up
    const bool ok = htp_kparams_solve(&sess, &op, &k);

    oracle_print_tensor(s); printf(" | "); oracle_print_tensor(d); printf(" | %u %zu =>", th, vt);
    print_trimmed(ok, &k, sizeof(k));
    n_cases++;
}

// A 2-d view with explicit strides, as the model's strided-copy primitive registers it: the
// third and fourth strides are the byte span.
ggml_tensor view2(ggml_type t, int64_t ne0, int64_t ne1, size_t nb0, size_t nb1) {
    const int64_t ne[4] = { ne0, ne1, 1, 1 };
    const size_t span = (ne0 > 0 ? ne0 - 1 : 0) * nb0 + (ne1 > 0 ? ne1 - 1 : 0) * nb1 + ggml_type_size(t);
    const size_t nb[4] = { nb0, nb1, span, span };
    return oracle_tensor_nb(t, ne, nb);
}

struct Shape { int64_t ne[4]; };

}  // namespace

int main() {
    const ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I32 };
    const Shape shapes[] = {
        { { 1, 1, 1, 1 } }, { { 0, 4, 1, 1 } }, { { 7, 1, 1, 1 } }, { { 64, 1, 1, 1 } }, { { 64, 5, 1, 1 } },
        { { 1024, 3, 1, 1 } }, { { 8, 4, 3, 2 } }, { { 3, 8, 2, 1 } }, { { 1024, 1, 1, 1 } }, { { 4096, 2, 1, 1 } },
        { { 2, 64, 1, 1 } }, { { 4, 33, 1, 1 } }, { { 16, 16, 1, 1 } }, { { 31, 17, 1, 1 } }, { { 32, 32, 1, 1 } },
        { { 33, 33, 1, 1 } }, { { 64, 64, 1, 1 } }, { { 65, 5, 1, 1 } }, { { 128, 16, 1, 1 } }, { { 2, 1024, 1, 1 } },
        { { 1, 256, 1, 1 } }, { { 5, 1, 1, 1 } }, { { 16, 8, 4, 1 } }, { { 32, 4, 2, 3 } }, { { 3, 5, 7, 2 } },
        { { 16384, 2, 1, 1 } }, { { 9, 6, 5, 1 } }, { { 1, 1, 5, 3 } }, { { 100, 8, 1, 1 } }, { { 300, 3, 1, 1 } },
        { { 1000, 16, 1, 1 } }, { { 2, 9000, 1, 1 } }, { { 4, 5000, 1, 1 } }, { { 8, 200, 1, 1 } },
    };
    // shapes that run the full session table (the ones whose kernel choice depends on it)
    const Shape session_shapes[] = {
        { { 64, 5, 1, 1 } }, { { 1024, 3, 1, 1 } }, { { 4096, 2, 1, 1 } }, { { 16384, 2, 1, 1 } }, { { 8, 4, 3, 2 } },
        { { 2, 1024, 1, 1 } }, { { 16, 16, 1, 1 } }, { { 64, 64, 1, 1 } }, { { 100, 8, 1, 1 } }, { { 2, 9000, 1, 1 } },
    };
    const uint32_t threads[] = { 0, 1, 6, 8 };
    const size_t   vtcms[]   = { 8u << 20, 64u << 10, 0 };

    auto is_session_shape = [&](const Shape & sh) {
        for (const Shape & s : session_shapes) if (!memcmp(s.ne, sh.ne, sizeof(sh.ne))) return true;
        return false;
    };

    // Same-extent pairs over every layout of each side. The default session for all of them; the
    // thread and budget table for the layouts that choose a threaded kernel (copy, transpose,
    // gather) on the shapes where that choice moves.
    for (ggml_type ts : types) for (ggml_type td : types) {
        const bool same = ts == td;
        const bool valid_convert = (ts == GGML_TYPE_F32 && td != GGML_TYPE_F32) || (td == GGML_TYPE_F32 && ts != GGML_TYPE_F32);
        for (const Shape & sh : shapes) {
            if (!same && (&sh - shapes) % 3 == 1 && !valid_convert) continue;  // the invalid pairs on a third of the shapes
            for (int ks = 0; ks < 5; ks++) for (int kd = 0; kd < 5; kd++) {
                if (!same && (ks > 3 || kd > 3 || (&sh - shapes) % 2 == 1)) continue;
                ggml_tensor s = make(ts, sh.ne, ks), d = make(td, sh.ne, kd);
                const bool full = is_session_shape(sh) && (same || valid_convert) && (ts != GGML_TYPE_I32 || td == GGML_TYPE_F32) &&
                                  ((kd == 0 && ks <= 4) || (ks == 0 && kd == 2) || (ks == 2 && kd == 2));
                for (uint32_t th : threads) for (size_t vt : vtcms) {
                    if (!full && (th != 6 || vt != (8u << 20))) continue;
                    emit(s, d, th, vt);
                }
            }
        }
    }

    // Same element count, different extents (a reshaping copy), over the layouts.
    const Shape reshaped[] = {
        { { 1, 7, 1, 1 } }, { { 8, 8, 1, 1 } }, { { 32, 2, 1, 1 } }, { { 4, 6, 1, 1 } }, { { 256, 4, 1, 1 } },
        { { 16, 64, 1, 1 } }, { { 1, 1024, 1, 1 } }, { { 2048, 1, 1, 1 } }, { { 4, 4, 4, 1 } }, { { 16, 4, 4, 4 } },
        { { 64, 16, 1, 1 } }, { { 128, 8, 1, 1 } }, { { 2, 512, 1, 1 } }, { { 12, 2, 1, 1 } },
    };
    for (ggml_type t : types) for (const Shape & a : shapes) for (const Shape & b : reshaped) {
        if (a.ne[0] * a.ne[1] * a.ne[2] * a.ne[3] != b.ne[0] * b.ne[1] * b.ne[2] * b.ne[3]) continue;
        if (memcmp(a.ne, b.ne, sizeof(a.ne)) == 0) continue;
        for (int ks = 0; ks < 5; ks++) for (int kd = 0; kd < 5; kd++) {
            ggml_tensor s = make(t, a.ne, ks), d = make(t, b.ne, kd);
            emit(s, d, 6, 8u << 20);
            emit(s, d, 0, 8u << 20);
            emit(s, d, 4, 64u << 10);
            ggml_tensor d2 = make(t == GGML_TYPE_F32 ? GGML_TYPE_F16 : GGML_TYPE_F32, b.ne, kd);
            emit(s, d2, 6, 8u << 20);
        }
    }

    // The strided 2-d copies the model issues: conv-state write-back and de-interleave, the
    // head-gate and q/gate packing, time-major <-> channel-major transposes, interleaved scatter.
    for (int64_t hs : { 96, 1024, 2048 }) {
        for (int64_t m : { 1, 2, 8, 32, 128 }) {
            for (int64_t slots : { 1, 2, 6 }) {
                for (uint32_t th : { 1u, 8u }) for (size_t vt : { size_t(8u << 20), size_t(64u << 10) }) {
                    // the last `slots` positions of a [slots + m, hs] window back into a [slots, hs] state
                    emit(view2(GGML_TYPE_F32, slots, hs, 4, (slots + m) * 4), view2(GGML_TYPE_F32, slots, hs, 4, slots * 4), th, vt);
                    // the new positions' [hs, m] tile into the channel-major window and back
                    emit(view2(GGML_TYPE_F32, m, hs, hs * 4, 4), view2(GGML_TYPE_F32, m, hs, 4, (slots + m) * 4), th, vt);
                    emit(view2(GGML_TYPE_F32, m, hs, 4, m * 4), view2(GGML_TYPE_F32, m, hs, hs * 4, 4), th, vt);
                }
            }
            // f16 versions of the transposes (K/V into the attention layout)
            emit(view2(GGML_TYPE_F16, m, hs, hs * 2, 2), view2(GGML_TYPE_F16, m, hs, 2, m * 2), 8, 8u << 20);
            emit(view2(GGML_TYPE_F16, hs, m, 2, hs * 4), view2(GGML_TYPE_F16, hs, m, 2, hs * 2), 8, 8u << 20);
        }
        for (int64_t nh : { 1, 8, 16 }) for (int64_t rows : { 1, 7, 32 }) {
            emit(view2(GGML_TYPE_F32, nh, rows, 4, hs * 4), view2(GGML_TYPE_F32, nh, rows, 4, nh * 4), 8, 8u << 20);
            emit(view2(GGML_TYPE_F32, 64, nh, 4, 2 * 64 * 4), view2(GGML_TYPE_F32, 64, nh, 4, 64 * 4), 8, 8u << 20);
        }
        // one interleaved state: both directions, one row
        emit(view2(GGML_TYPE_F32, hs, 1, 8, 8), view2(GGML_TYPE_F32, hs, 1, 4, hs * 4), 8, 8u << 20);
        emit(view2(GGML_TYPE_F32, hs, 1, 4, hs * 4), view2(GGML_TYPE_F32, hs, 1, 8, 8), 8, 8u << 20);
        emit(view2(GGML_TYPE_F32, hs, 1, 8, 8), view2(GGML_TYPE_F32, hs, 1, 4, hs * 4), 0, 0);
    }

    // Pairs the copy kernels refuse: element counts that differ, types outside F32/F16/I32,
    // and conversions without a kernel.
    {
        const int64_t a[4] = { 4, 4, 1, 1 }, b[4] = { 4, 3, 1, 1 };
        emit(make(GGML_TYPE_F32, a, 0), make(GGML_TYPE_F32, b, 0), 4, 8u << 20);
        const ggml_type exotic[] = { GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_I64, GGML_TYPE_F64, GGML_TYPE_I8 };
        const int64_t e[4] = { 64, 4, 1, 1 };
        for (ggml_type t : exotic) {
            emit(oracle_tensor(t, e, 0), oracle_tensor(GGML_TYPE_F32, e, 0), 4, 8u << 20);
            emit(oracle_tensor(GGML_TYPE_F32, e, 0), oracle_tensor(t, e, 0), 4, 8u << 20);
            emit(oracle_tensor(t, e, 0), oracle_tensor(t, e, 0), 4, 8u << 20);
        }
        const int64_t one[4] = { 1, 1, 1, 1 };
        for (ggml_type s : types) for (ggml_type d : types) {
            emit(oracle_tensor(s, one, 0), oracle_tensor(d, one, 0), 4, 8u << 20);  // scalar copies
            emit(oracle_tensor(s, e, 0), oracle_tensor(d, e, 0), 4, 8u << 20);
        }
    }

    fprintf(stderr, "%d cases\n", n_cases);
    return 0;
}
