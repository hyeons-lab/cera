// Case grid for htp_mm_hmx_compute_chunks (matmul-ops.h), the HMX (M, N) chunk search; one line per
// case: <vtcm> <overhead> <per_n> <per_m> <per_mn> <m> <n> <m_block_cost> <n_block_cost> => <rc> <mc> <nc> <total>
#include "common.h"

int main() {
    const size_t vtcms[]   = { 8u << 20, 2u << 20, 1u << 20, 512u << 10 };
    const size_t per_ns[]  = { 640, 1280, 4096, 7168 };
    const size_t per_ms[]  = { 1024, 4096, 16384 };
    const size_t per_mns[] = { 2, 4 };
    const size_t ms[]      = { 32, 64, 96, 512, 2048 };
    // N includes values that are not multiples of the chosen chunk, so the tail-waste tie-break fires
    const size_t ns[]      = { 32, 96, 1024, 1536, 2048, 2880, 4096, 6144, 9216 };
    const size_t costs[][2] = { { 300, 500 }, { 800, 200 }, { 1, 1 }, { 0, 7 } };
    const size_t overheads[] = { 4 * 2048 + 256, 7 * 2048 + 256 };
    int n = 0;
    for (size_t vt : vtcms) for (size_t ov : overheads) for (size_t pn : per_ns) for (size_t pm : per_ms)
    for (size_t pmn : per_mns) for (size_t m : ms) for (size_t nn : ns) for (auto & c : costs) {
        if (ov != overheads[0] && (pm != per_ms[0] || m == 96)) continue;  // the pipelined overhead on a thinner grid
        size_t mc = 0, nc = 0, total = 0;
        const int rc = htp_mm_hmx_compute_chunks(vt, ov, pn, pm, pmn, m, nn, c[0], c[1], &mc, &nc, &total);
        printf("%zu %zu %zu %zu %zu %zu %zu %zu %zu => %d %zu %zu %zu\n", vt, ov, pn, pm, pmn, m, nn, c[0], c[1], rc,
               rc == 0 ? mc : 0, rc == 0 ? nc : 0, rc == 0 ? total : 0);
        n++;
    }
    // degenerate inputs the search refuses
    const size_t bad[][9] = { { 8u << 20, 8448, 640, 1024, 2, 0, 1024, 1, 1 }, { 8u << 20, 8448, 640, 1024, 2, 64, 0, 1, 1 },
                              { 8448, 8448, 640, 1024, 2, 64, 1024, 1, 1 }, { 8u << 20, 8448, 0, 1024, 2, 64, 1024, 1, 1 },
                              { 8u << 20, 8448, 640, 0, 2, 64, 1024, 1, 1 }, { 8u << 20, 8448, 640, 1024, 0, 64, 1024, 1, 1 },
                              { 8u << 20, 8448, 1u << 30, 1024, 2, 64, 1024, 1, 1 } };
    for (auto & b : bad) {
        size_t mc = 0, nc = 0, total = 0;
        const int rc = htp_mm_hmx_compute_chunks(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], &mc, &nc, &total);
        printf("%zu %zu %zu %zu %zu %zu %zu %zu %zu => %d %zu %zu %zu\n", b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], rc,
               rc == 0 ? mc : 0, rc == 0 ? nc : 0, rc == 0 ? total : 0);
        n++;
    }
    fprintf(stderr, "%d cases\n", n);
    return 0;
}
