// Case table for ggml_hexagon_precompute_cpy_params; prints one line per case.
#include <stdio.h>
#include <stdlib.h>

#include "fn.inc"

struct Shape { int64_t ne[4]; };

// kind: 0 contiguous, 1 row-padded, 2 transposed view, 3 outer dims permuted
static ggml_tensor make(ggml_type t, const int64_t * ne, int kind) {
    ggml_tensor x = {};
    x.type = t;
    const size_t sz = ggml_type_size(t);
    for (int i = 0; i < 4; i++) x.ne[i] = ne[i];
    switch (kind) {
        case 0:
            x.nb[0] = sz; x.nb[1] = sz * ne[0]; x.nb[2] = x.nb[1] * ne[1]; x.nb[3] = x.nb[2] * ne[2];
            break;
        case 1:
            x.nb[0] = sz; x.nb[1] = sz * ne[0] + 16; x.nb[2] = x.nb[1] * ne[1]; x.nb[3] = x.nb[2] * ne[2];
            break;
        case 2:  // view of the transposed [ne1, ne0] matrix
            x.nb[0] = sz * ne[1]; x.nb[1] = sz; x.nb[2] = sz * ne[0] * ne[1]; x.nb[3] = x.nb[2] * ne[2];
            break;
        case 3:  // dim 2 stored inside dim 1
            x.nb[0] = sz; x.nb[2] = sz * ne[0]; x.nb[1] = x.nb[2] * ne[2]; x.nb[3] = x.nb[1] * ne[1];
            break;
    }
    return x;
}

static void print_tensor(const ggml_tensor & t) {
    printf("%d %lld %lld %lld %lld %zu %zu %zu %zu", (int) t.type, (long long) t.ne[0], (long long) t.ne[1],
           (long long) t.ne[2], (long long) t.ne[3], t.nb[0], t.nb[1], t.nb[2], t.nb[3]);
}

int main() {
    const ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I32 };
    const Shape shapes[] = { { { 1, 1, 1, 1 } }, { { 0, 4, 1, 1 } }, { { 7, 1, 1, 1 } }, { { 64, 1, 1, 1 } },
                             { { 64, 5, 1, 1 } }, { { 1024, 3, 1, 1 } }, { { 8, 4, 3, 2 } }, { { 3, 8, 2, 1 } },
                             { { 1024, 1, 1, 1 } }, { { 4096, 2, 1, 1 } } };
    const Shape reshaped[] = { { { 1, 7, 1, 1 } }, { { 8, 8, 1, 1 } }, { { 32, 2, 1, 1 } }, { { 4, 6, 1, 1 } } };
    const int threads[] = { 0, 4, 6 };
    const size_t vtcms[] = { 8u << 20, 64u << 10, 0 };

    int n = 0;
    auto emit = [&](const ggml_tensor & s, const ggml_tensor & d, int th, size_t vt) {
        ggml_hexagon_session sess = { th, vt };
        ggml_tensor op = d;
        op.src[0] = const_cast<ggml_tensor *>(&s);
        htp_copy_kernel_params k;
        memset(&k, 0xAA, sizeof(k));   // poison, so a field the function forgets to set shows up
        const bool ok = ggml_hexagon_precompute_cpy_params(&sess, &op, &k);
        print_tensor(s); printf(" | "); print_tensor(d); printf(" | %d %zu =>", th, vt);
        if (!ok) { printf(" unsupported\n"); n++; return; }
        uint32_t w[32] = {};
        memcpy(w, &k, sizeof(k) < sizeof(w) ? sizeof(k) : sizeof(w));
        for (int i = 0; i < 32; i++) printf(" %u", w[i]);
        printf("\n");
        n++;
    };

    for (ggml_type ts : types) for (ggml_type td : types)
    for (const Shape & sh : shapes) for (int ks = 0; ks < 4; ks++) for (int kd = 0; kd < 4; kd++) {
        // keep the table a few thousand lines: threads/vtcm only for the contiguous kinds
        const bool plain = (ks == 0 && kd == 0);
        for (int th : threads) for (size_t vt : vtcms) {
            if (!plain && !(th == 4 && vt == (8u << 20))) continue;
            ggml_tensor s = make(ts, sh.ne, ks), d = make(td, sh.ne, kd);
            emit(s, d, th, vt);
        }
    }
    // same element count, different extents (reshape), several strides
    for (ggml_type t : types) for (const Shape & a : shapes) for (const Shape & b : reshaped) {
        if (a.ne[0] * a.ne[1] * a.ne[2] * a.ne[3] != b.ne[0] * b.ne[1] * b.ne[2] * b.ne[3]) continue;
        for (int ks = 0; ks < 4; ks++) for (int kd = 0; kd < 4; kd++) {
            ggml_tensor s = make(t, a.ne, ks), d = make(t, b.ne, kd);
            emit(s, d, 4, 8u << 20);
            ggml_tensor d2 = make(t == GGML_TYPE_F32 ? GGML_TYPE_F16 : GGML_TYPE_F32, b.ne, kd);
            emit(s, d2, 6, 8u << 20);
        }
    }
    // element counts that differ
    { int64_t a[4] = {4, 4, 1, 1}, b[4] = {4, 3, 1, 1}; ggml_tensor s = make(GGML_TYPE_F32, a, 0), d = make(GGML_TYPE_F32, b, 0); emit(s, d, 4, 8u << 20); }
    fprintf(stderr, "%d cases\n", n);
    return 0;
}
