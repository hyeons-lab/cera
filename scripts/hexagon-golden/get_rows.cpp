// Case table for ggml_hexagon_precompute_get_rows_params; prints one line per case.
#include <stdio.h>
#include <stdlib.h>

#include "fn.inc"

int main() {
    const ggml_type srcs[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0 };
    const ggml_type dsts[] = { GGML_TYPE_F32, GGML_TYPE_F16 };
    const int64_t ne00s[] = { 1, 32, 64, 1000, 2048, 4096 };
    const int64_t idx[][3] = { { 1, 1, 1 }, { 4, 1, 1 }, { 7, 3, 1 }, { 64, 2, 2 }, { 0, 1, 1 } };
    const int threads[] = { 0, 1, 4, 6, 8 };
    const size_t vtcms[] = { 8u << 20, 4096, 100000, 0 };
    int n = 0;
    for (ggml_type ts : srcs) for (ggml_type td : dsts) for (int64_t ne00 : ne00s) for (auto & ix : idx)
    for (int tiled = 0; tiled < 2; tiled++) for (int th : threads) for (size_t vt : vtcms) {
        // keep the table moderate: vary threads/vtcm fully only for a few shapes
        if ((th != 4 && th != 0) && vt != (8u << 20) && ne00 != 64) continue;
        ggml_tensor s0 = {}, s1 = {}, d = {};
        s0.type = ts; s0.ne[0] = ne00; s0.ne[1] = 100; s0.ne[2] = 2; s0.ne[3] = 3;
        s1.type = GGML_TYPE_I32; s1.ne[0] = ix[0]; s1.ne[1] = ix[1]; s1.ne[2] = ix[2]; s1.ne[3] = 1;
        d.type = td;
        ggml_hexagon_session sess = { th, vt, {} };
        if (tiled) sess.needs_repack.insert(&s0);
        htp_get_rows_kernel_params k;
        memset(&k, 0xAA, sizeof(k));
        ggml_hexagon_precompute_get_rows_params(&sess, &s0, &s1, &d, &k);
        printf("%d %d %lld %lld %lld %lld %lld %lld %d | %d %zu =>", (int) ts, (int) td, (long long) ne00, (long long) s0.ne[2],
               (long long) s0.ne[3], (long long) s1.ne[0], (long long) s1.ne[1], (long long) s1.ne[2], tiled, th, vt);
        uint32_t w[17] = {};
        memcpy(w, &k, sizeof(k) < sizeof(w) ? sizeof(k) : sizeof(w));
        for (int i = 0; i < 17; i++) printf(" %u", w[i]);
        printf("\n");
        n++;
    }
    fprintf(stderr, "%d cases\n", n);
    return 0;
}
