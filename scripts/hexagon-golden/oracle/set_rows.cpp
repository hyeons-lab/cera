// Case table for SET_ROWS through the real `htp_kparams_solve`. Line format:
//   <values> | <indices> | <dst> | <session threads> <session vtcm bytes> => <32 words | unsupported>
#include "common.h"

int main() {
    const ggml_type dst_types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 };
    const ggml_type src_types[] = { GGML_TYPE_F32, GGML_TYPE_F16 };
    const ggml_type idx_types[] = { GGML_TYPE_I32, GGML_TYPE_I64, GGML_TYPE_F32 };
    const int64_t widths[]      = { 1, 31, 32, 64, 128, 512, 1024, 2048, 8192, 65536 };
    const int64_t rows[]        = { 1, 3, 17, 512 };
    const int64_t heads[]       = { 1, 8 };
    const uint32_t threads[]    = { 0, 1, 4, 6, 8 };
    const size_t vtcms[]        = { 8u << 20, 1u << 20, 64u << 10, 0 };

    int n = 0;
    auto emit = [&](ggml_type st, ggml_type it, ggml_type dt, int64_t ne00, int64_t nr, int64_t nh,
                    uint32_t th, size_t vt) {
        const int64_t s_ne[4] = { ne00, nr, nh, 1 };
        const int64_t i_ne[4] = { nr, 1, 1, 1 };
        const int64_t d_ne[4] = { ne00, 64, nh, 1 };
        ggml_tensor s = oracle_tensor(st, s_ne), i = oracle_tensor(it, i_ne), d = oracle_tensor(dt, d_ne);

        ggml_hexagon_session sess;
        sess.n_threads = th;
        sess.vtcm_size = vt;

        ggml_tensor op = d;
        op.op     = GGML_OP_SET_ROWS;
        op.src[0] = &s;
        op.src[1] = &i;
        htp_set_rows_kernel_params k;
        memset(&k, 0xAA, sizeof(k));  // poison, so a field the solver forgets to set shows up
        const bool ok = htp_kparams_solve(&sess, &op, &k);

        oracle_print_tensor(s); printf(" | "); oracle_print_tensor(i); printf(" | "); oracle_print_tensor(d);
        printf(" | %u %zu =>", th, vt);
        oracle_print_words(ok, &k, sizeof(k));
        n++;
    };

    for (ggml_type dt : dst_types) for (int64_t w : widths) {
        if (dt == GGML_TYPE_Q8_0 && w >= 32 && w % 32 != 0) continue;  // not a whole number of blocks
        if (dt == GGML_TYPE_Q4_0 && w % 32 != 0) continue;
        for (int64_t nr : rows) for (int64_t nh : heads) for (uint32_t th : threads) for (size_t vt : vtcms) {
            // the full cross product is large and mostly redundant: keep every thread count and
            // budget for the common index type, and the other types once
            emit(GGML_TYPE_F32, GGML_TYPE_I32, dt, w, nr, nh, th, vt);
        }
    }
    for (ggml_type st : src_types) for (ggml_type it : idx_types) for (ggml_type dt : dst_types) {
        emit(st, it, dt, 128, 4, 1, 6, 8u << 20);
    }
    fprintf(stderr, "%d cases\n", n);
    return 0;
}
