// Case table for ggml_hexagon_precompute_ssm_conv_params; prints one line per case.
#include <stdio.h>
#include <stdlib.h>

#include "fn.inc"

int main() {
    const int64_t d_convs[] = { 3, 4 };
    const int64_t d_inners[] = { 1, 31, 32, 33, 96, 1024, 2048, 8192 };
    const int64_t n_ts[] = { 1, 2, 16, 32, 1024 };
    const int64_t n_ss[] = { 1, 2 };
    const int threads[] = { 1, 2, 4, 6, 8 };
    const size_t vtcms[] = { 8u << 20, 1u << 20, 64u << 10, 0 };
    int n = 0;
    for (int64_t dc : d_convs) for (int64_t di : d_inners) for (int64_t nt : n_ts) for (int64_t ns : n_ss)
    for (int th : threads) for (size_t vt : vtcms) {
        ggml_tensor s0 = {}, s1 = {}, d = {};
        s0.ne[0] = dc - 1 + nt; s0.ne[1] = di; s0.ne[2] = ns; s0.ne[3] = 1;
        s1.ne[0] = dc; s1.ne[1] = di; s1.ne[2] = 1; s1.ne[3] = 1;
        d.ne[0] = di; d.ne[1] = nt; d.ne[2] = ns; d.ne[3] = 1;
        ggml_hexagon_session sess = { th, vt, {} };
        htp_ssm_conv_kernel_params k;
        memset(&k, 0xAA, sizeof(k));
        ggml_hexagon_precompute_ssm_conv_params(&sess, &s0, &s1, &d, &k);
        printf("%lld %lld %lld %lld %lld | %d %zu =>", (long long) dc, (long long) di, (long long) nt, (long long) ns,
               (long long) s0.ne[0], th, vt);
        uint32_t w[13] = {};
        memcpy(w, &k, sizeof(k) < sizeof(w) ? sizeof(k) : sizeof(w));
        for (int i = 0; i < 13; i++) printf(" %u", w[i]);
        printf("\n");
        n++;
    }
    fprintf(stderr, "%d cases\n", n);
    return 0;
}
