// Case grid for the HMX chunk solvers of matmul-ops.h, one line per case.
//   2d      <wtype> <k> <m_id_rows> <ne01_padded> <ne11_padded> <m_for_cost> <threads> <pipeline> <is_id> <aligned_tile> <bias> <vtcm>
//             => <ok> <m_chunk> <n_chunk> <act_threads> <vtcm_used>        (htp_mm_hmx_solve_2d_params)
//   batched <k> <ne01_padded> <ne11> <group> <threads> <pipeline> <bias> <vtcm>
//             => <ok> <m_chunk> <n_chunk> <act_threads> <vtcm_used>        (htp_mm_hmx_solve_batched_params)
// Ragged K/N, odd thread counts and budgets down to 64 KiB reach the no-fit and act-thread-halving paths.
#include "common.h"

static int n_cases = 0;

static void two_d(int wt, uint32_t k, uint32_t m_id_rows, uint32_t nn, uint32_t m_pad, uint32_t m_cost, int th,
                  bool pipeline, bool is_id, uint32_t aligned, size_t bias, size_t vt) {
    size_t mc = 0, nc = 0, vtcm = 0;
    int act = 0;
    const bool ok = htp_mm_hmx_solve_2d_params(wt, k, m_id_rows, nn, m_pad, m_cost, th, pipeline, is_id, aligned, bias, vt,
                                               &mc, &nc, &act, &vtcm);
    printf("2d %d %u %u %u %u %u %d %d %d %u %zu %zu => %d %zu %zu %d %zu\n", wt, k, m_id_rows, nn, m_pad, m_cost, th,
           pipeline ? 1 : 0, is_id ? 1 : 0, aligned, bias, vt, ok ? 1 : 0, ok ? mc : 0, ok ? nc : 0, ok ? act : 0,
           ok ? vtcm : 0);
    n_cases++;
}

static void batched(uint32_t k, uint32_t nn, uint32_t m, uint32_t group, int th, bool pipeline, size_t bias, size_t vt) {
    size_t mc = 0, nc = 0, vtcm = 0;
    int act = 0;
    const bool ok = htp_mm_hmx_solve_batched_params(HTP_TYPE_F16, k, nn, m, group, th, pipeline, bias, vt, &mc, &nc, &act, &vtcm);
    printf("batched %u %u %u %u %d %d %zu %zu => %d %zu %zu %d %zu\n", k, nn, m, group, th, pipeline ? 1 : 0, bias, vt,
           ok ? 1 : 0, ok ? mc : 0, ok ? nc : 0, ok ? act : 0, ok ? vtcm : 0);
    n_cases++;
}

int main() {
    const int wtypes[]       = { HTP_TYPE_Q4_0, HTP_TYPE_Q8_0, HTP_TYPE_Q4_K, HTP_TYPE_Q6_K };
    const uint32_t ks[]      = { 512, 768, 1024, 1280, 2560, 4096 };
    const uint32_t ns[]      = { 32, 96, 1024, 2880, 4320, 8192, 12288 };
    const uint32_t ms[]      = { 5, 32, 100, 2048 };
    const int threads[]      = { 1, 3, 4, 8 };
    const size_t budgets[]   = { 8u << 20, 2u << 20, 512u << 10, 224u << 10, 192u << 10, 160u << 10, 128u << 10, 96u << 10, 64u << 10 };

    // the non-ID tiled solve as `htp_kparams_solve_hmx_mm` calls it for a plain matmul
    for (int wt : wtypes) for (uint32_t k : ks) for (uint32_t nn : ns) for (uint32_t m : ms)
    for (int th : threads) for (size_t vt : budgets) {
        const uint32_t m_pad = (m + 31) / 32 * 32;
        two_d(wt, k, 0, nn, m_pad, m, th, htp_mm_hmx_pipeline(m), false, htp_tensor_get_tile_padded_size(wt), 0, vt);
    }
    // plain F16 / F32 weights (no tiles: aligned tile size 0)
    for (int wt : { HTP_TYPE_F16, HTP_TYPE_F32 }) for (uint32_t k : { 512, 1024, 4096 }) for (uint32_t nn : { 96, 1024, 4608 })
    for (uint32_t m : { 5, 33, 200 }) for (int th : { 1, 8 }) for (size_t vt : { 8u << 20, 1u << 20, 256u << 10, 64u << 10 }) {
        two_d(wt, k, 0, nn, (m + 31) / 32 * 32, m, th, htp_mm_hmx_pipeline(m), false, 0, 0, vt);
    }
    // a fused bias (MUL_MAT_ADD) takes VTCM too
    for (int wt : { HTP_TYPE_Q8_0, HTP_TYPE_Q4_0 }) for (uint32_t k : { 1024, 4096 }) for (uint32_t nn : { 1024, 4608 })
    for (uint32_t m : { 5, 33, 130 }) for (size_t bias : { 128, 4096, 65536 }) for (int th : { 4, 8 })
    for (size_t vt : { 8u << 20, 1u << 20, 256u << 10, 128u << 10 }) {
        two_d(wt, k, 0, nn, (m + 31) / 32 * 32, m, th, htp_mm_hmx_pipeline(m), false, htp_tensor_get_tile_padded_size(wt), bias, vt);
    }
    // routed experts: rows per expert, no activation threads
    for (int wt : wtypes) for (uint32_t k : { 512, 1024, 4096 }) for (uint32_t nn : { 96, 1024, 4608 })
    for (uint32_t rows : { 32, 64, 256 }) for (int th : { 1, 8 }) for (size_t vt : { 8u << 20, 1u << 20, 256u << 10, 128u << 10, 64u << 10 }) {
        two_d(wt, k, rows, nn, 32, rows, th, false, true, htp_tensor_get_tile_padded_size(wt), 0, vt);
    }
    // the grouped (GQA) batched solve
    for (uint32_t k : { 64, 128, 576 }) for (uint32_t nn : { 64, 128, 512 }) for (uint32_t m : { 5, 32, 100 })
    for (uint32_t g : { 2, 4, 8 }) for (int th : { 1, 8 }) for (size_t vt : { 8u << 20, 1u << 20, 256u << 10, 64u << 10 }) {
        batched(k, nn, m, g, th, htp_mm_hmx_pipeline(m), 0, vt);
        if (m == 32 && th == 8) batched(k, nn, m, g, th, false, 4096, vt);
    }

    fprintf(stderr, "%d cases\n", n_cases);
    return 0;
}
