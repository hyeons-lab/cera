// Case table for ggml_hexagon_precompute_concat_params; prints one line per case.
#include <stdio.h>
#include <stdlib.h>

#include "fn.inc"

// kind: 0 contiguous, 1 row-padded, 2 transposed view of the [ne1, ne0] matrix
static ggml_tensor make(ggml_type t, const int64_t * ne, int kind) {
    ggml_tensor x = {};
    x.type = t;
    const size_t sz = ggml_type_size(t);
    for (int i = 0; i < 4; i++) x.ne[i] = ne[i];
    if (kind == 2) {
        x.nb[0] = sz * ne[1]; x.nb[1] = sz; x.nb[2] = sz * ne[0] * ne[1]; x.nb[3] = x.nb[2] * ne[2];
    } else {
        x.nb[0] = sz; x.nb[1] = sz * ne[0] + (kind == 1 ? 16 : 0); x.nb[2] = x.nb[1] * ne[1]; x.nb[3] = x.nb[2] * ne[2];
    }
    return x;
}

static void print_tensor(const ggml_tensor & t) {
    printf("%d %lld %lld %lld %lld %zu %zu %zu %zu", (int) t.type, (long long) t.ne[0], (long long) t.ne[1],
           (long long) t.ne[2], (long long) t.ne[3], t.nb[0], t.nb[1], t.nb[2], t.nb[3]);
}

int main() {
    const ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I32 };
    const int64_t bases[][4] = { { 6, 4, 1, 1 }, { 64, 3, 2, 1 }, { 1, 1, 1, 1 }, { 33, 7, 1, 1 }, { 256, 16, 1, 1 } };
    const int threads[] = { 0, 4, 8 };
    const size_t vtcms[] = { 8u << 20, 32u << 10, 0 };
    int n = 0;
    auto emit = [&](const ggml_tensor & a, const ggml_tensor & b, const ggml_tensor & d, int dim, int th, size_t vt) {
        ggml_hexagon_session sess = { th, vt };
        ggml_tensor op = d;
        op.src[0] = const_cast<ggml_tensor *>(&a);
        op.src[1] = const_cast<ggml_tensor *>(&b);
        op.op_params[0] = dim;
        htp_concat_kernel_params k;
        memset(&k, 0xAA, sizeof(k));
        const bool ok = ggml_hexagon_precompute_concat_params(&sess, &op, &k);
        print_tensor(a); printf(" | "); print_tensor(b); printf(" | "); print_tensor(d);
        printf(" | %d %d %zu =>", dim, th, vt);
        if (!ok) { printf(" unsupported\n"); n++; return; }
        uint32_t w[8] = {};
        memcpy(w, &k, sizeof(k) < sizeof(w) ? sizeof(k) : sizeof(w));
        for (int i = 0; i < 8; i++) printf(" %u", w[i]);
        printf("\n");
        n++;
    };
    for (ggml_type t : types) for (const auto & base : bases) for (int dim = -1; dim <= 4; dim++) {
        const int dd = dim < 0 || dim > 3 ? 0 : dim;
        int64_t ea[4], eb[4], ed[4];
        for (int i = 0; i < 4; i++) { ea[i] = eb[i] = ed[i] = base[i]; }
        eb[dd] = base[dd] + 3;            // src1 longer along dim
        ed[dd] = ea[dd] + eb[dd];
        for (int ka = 0; ka < 3; ka++) for (int kb = 0; kb < 3; kb++) for (int kd = 0; kd < 2; kd++) {
            ggml_tensor a = make(t, ea, ka), b = make(t, eb, kb), d = make(t, ed, kd);
            for (int th : threads) for (size_t vt : vtcms) {
                if ((th != 4 || vt != (8u << 20)) && !(ka == 0 && kb == 2 && kd == 0)) continue;
                emit(a, b, d, dim, th, vt);
            }
        }
    }
    // mismatching dst extents and mixed types
    { int64_t ea[4] = {6,4,1,1}, eb[4] = {2,4,1,1}, ed[4] = {7,4,1,1};
      ggml_tensor a = make(GGML_TYPE_F32, ea, 0), b = make(GGML_TYPE_F32, eb, 0), d = make(GGML_TYPE_F32, ed, 0); emit(a, b, d, 0, 4, 8u << 20);
      ggml_tensor h = make(GGML_TYPE_F16, eb, 0); emit(a, h, d, 0, 4, 8u << 20); }
    fprintf(stderr, "%d cases\n", n);
    return 0;
}
