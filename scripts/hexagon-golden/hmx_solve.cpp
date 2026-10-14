// Case grid for htp_mm_hmx_solve_2d_params (matmul-ops.h), the non-ID HMX 2-D matmul solver; one line per case.
// Ragged K/N, odd thread counts and budgets down to 64 KiB reach the no-fit and act-thread-halving paths.
#include <stdio.h>
#include <stdlib.h>

int main() {
    const int wtypes[]       = { HTP_TYPE_Q4_0, HTP_TYPE_Q8_0, HTP_TYPE_Q4_K, HTP_TYPE_Q6_K };
    const uint32_t ks[]      = { 512, 768, 1024, 1280, 2560, 4096 };
    const uint32_t ns[]      = { 32, 96, 1024, 2880, 4320, 8192, 12288 };
    const uint32_t ms[]      = { 5, 32, 100, 2048 };
    const int threads[]      = { 1, 3, 4, 8 };
    const size_t budgets[]   = { 8u << 20, 2u << 20, 512u << 10, 224u << 10, 192u << 10, 160u << 10, 128u << 10, 96u << 10, 64u << 10 };
    int n = 0;
    for (int wt : wtypes) for (uint32_t k : ks) for (uint32_t nn : ns) for (uint32_t m : ms)
    for (int th : threads) for (size_t vt : budgets) {
        const uint32_t m_pad = (m + 31) / 32 * 32;
        const bool pipeline  = htp_mm_hmx_pipeline(m);
        const uint32_t aligned = htp_mm_get_weight_aligned_tile_size(wt);
        size_t mc = 0, nc = 0, vtcm = 0;
        int act = 0;
        const bool ok = htp_mm_hmx_solve_2d_params(wt, k, 0, nn, m_pad, m, th, pipeline, false, aligned, 0, vt, &mc, &nc, &act, &vtcm);
        printf("%d %u %u %u %u %d %zu => %d %zu %zu %d %zu\n", wt, k, nn, m_pad, m, th, vt, ok ? 1 : 0, ok ? mc : 0, ok ? nc : 0,
               ok ? act : 0, ok ? vtcm : 0);
        n++;
    }
    fprintf(stderr, "%d cases\n", n);
    return 0;
}
