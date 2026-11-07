// Case table for GET_ROWS through the real `htp_kparams_solve`. Line format:
//   <table> <tiled> | <indices> | <dst> | <session threads> <session vtcm bytes> => <12 words | unsupported>
// `tiled` is 1 when the table is in the DSP's tiled layout (the host's flag on the tensor). The words
// are `htp_get_rows_kernel_params`: kernel_type, n_threads, total_tasks, vtcm_size, then div_ne10,
// div_ne10_ne11, div_ne02, div_ne03 {mp, l}.
#include "common.h"

namespace {

int n_cases = 0;

void emit(const ggml_tensor & s0, bool tiled, const ggml_tensor & s1, const ggml_tensor & d, uint32_t th, size_t vt) {
    ggml_hexagon_session sess;
    sess.n_threads = th;
    sess.vtcm_size = vt;

    ggml_tensor table = s0;
    if (tiled) {
        table.extra = (void *) (uintptr_t) HTP_TENSOR_TILED;
    }
    ggml_tensor op = d;
    op.op     = GGML_OP_GET_ROWS;
    op.src[0] = &table;
    op.src[1] = const_cast<ggml_tensor *>(&s1);

    htp_get_rows_kernel_params k;
    memset(&k, 0xAA, sizeof(k));
    const bool ok = htp_kparams_solve(&sess, &op, &k);

    oracle_print_tensor(s0); printf(" %d | ", (int) tiled); oracle_print_tensor(s1); printf(" | "); oracle_print_tensor(d);
    printf(" | %u %zu =>", th, vt);
    oracle_print_words(ok, &k, sizeof(k), sizeof(k) / 4);
    n_cases++;
}

// table [ne00, 100, 2, 3] gathered by indices [ne10, ne11, ne12] into dst [ne00, ne10, ne11, ne12].
void emit_shape(ggml_type st, ggml_type dt, ggml_type it, int64_t ne00, const int64_t * ix, bool tiled, uint32_t th,
                size_t vt, int skind = 0, int dkind = 0) {
    const int64_t s_ne[4] = { ne00, 100, 2, 3 };
    const int64_t i_ne[4] = { ix[0], ix[1], ix[2], 1 };
    const int64_t d_ne[4] = { ne00, ix[0], ix[1], ix[2] };
    ggml_tensor s = oracle_tensor(st, s_ne, skind);
    ggml_tensor i = oracle_tensor(it, i_ne);
    ggml_tensor d = oracle_tensor(dt, d_ne, dkind);
    emit(s, tiled, i, d, th, vt);
}

struct kind_widths {
    ggml_type type;
    std::initializer_list<int64_t> widths;
    bool has_tile;
};

}  // namespace

int main() {
    const uint32_t threads[] = { 0, 1, 4, 6, 8 };
    const size_t   vtcms[]   = { 8u << 20, 4096, 100000, 0 };
    const int64_t  idx_shapes[][3] = { { 1, 1, 1 }, { 4, 1, 1 }, { 7, 3, 1 }, { 64, 2, 2 }, { 0, 1, 1 }, { 512, 1, 1 },
                                       { 1, 512, 1 }, { 33, 1, 1 }, { 128, 1, 1 }, { 129, 1, 1 } };

    // The tables the solver takes: dense, quantized (flat and tiled), by supported index type.
    const kind_widths tables[] = {
        { GGML_TYPE_F32,  { 1, 32, 64, 1000, 4096 }, true },
        { GGML_TYPE_F16,  { 1, 32, 64, 1000, 4096 }, true },
        { GGML_TYPE_BF16, { 32, 64, 1000 }, true },
        { GGML_TYPE_I32,  { 1, 32, 64, 1000 }, true },
        { GGML_TYPE_Q4_0, { 32, 64, 1024, 4096 }, true },
        { GGML_TYPE_Q8_0, { 32, 64, 1024, 4096 }, true },
        { GGML_TYPE_Q4_K, { 256, 512, 4096 }, true },
        { GGML_TYPE_Q5_K, { 256, 512, 4096 }, true },
        { GGML_TYPE_Q6_K, { 256, 512, 4096 }, true },
        { GGML_TYPE_Q1_0, { 128, 256, 2048 }, true },
        { GGML_TYPE_Q2_0, { 64, 128, 2048 }, true },
    };

    for (const kind_widths & t : tables) {
        // the output is F32 for every table, and the table's own type where a same-type gather exists
        std::initializer_list<ggml_type> dsts = { GGML_TYPE_F32, t.type };
        for (ggml_type dt : dsts) {
            if (dt == t.type && t.type != GGML_TYPE_F16 && t.type != GGML_TYPE_I32 && t.type != GGML_TYPE_F32) continue;
            if (dt == t.type && t.type == GGML_TYPE_F32) continue;  // already the first entry
            for (int64_t w : t.widths) for (const int64_t * ix : idx_shapes) for (int tiled = 0; tiled < 2; tiled++) {
                // the common settings for every shape ...
                emit_shape(t.type, dt, GGML_TYPE_I32, w, ix, tiled, 6, 8u << 20);
                emit_shape(t.type, dt, GGML_TYPE_I32, w, ix, tiled, 4, 4096);
            }
            // ... and the full thread and budget grid on the shapes the models use
            const int64_t w_grid[] = { *t.widths.begin(), *(t.widths.end() - 1) };
            for (int64_t w : w_grid) for (int tiled = 0; tiled < 2; tiled++) {
                for (const int64_t * ix : { idx_shapes[2], idx_shapes[3], idx_shapes[0] }) {
                    for (uint32_t th : threads) for (size_t vt : vtcms) {
                        emit_shape(t.type, dt, GGML_TYPE_I32, w, ix, tiled, th, vt);
                    }
                }
            }
        }
    }

    // I64 indices: the buffer for them is twice as large.
    for (const kind_widths & t : tables) for (int64_t w : t.widths) for (const int64_t * ix : { idx_shapes[1], idx_shapes[3], idx_shapes[5], idx_shapes[9] })
    for (int tiled = 0; tiled < 2; tiled++) for (uint32_t th : { 1u, 6u }) for (size_t vt : { (size_t) (8u << 20), (size_t) 4096 }) {
        const ggml_type dt = (t.type == GGML_TYPE_I32) ? GGML_TYPE_I32 : GGML_TYPE_F32;
        emit_shape(t.type, dt, GGML_TYPE_I64, w, ix, tiled, th, vt);
    }

    // The scalar gather cera sends: an F32 table of single values, one I32 index per output.
    for (int64_t n_in : { 1, 8, 64 }) for (int64_t n_out : { 1, 4, 8, 129 })
    for (uint32_t th : threads) for (size_t vt : vtcms) {
        const int64_t s_ne[4] = { 1, n_in, 1, 1 };
        const int64_t i_ne[4] = { n_out, 1, 1, 1 };
        const int64_t d_ne[4] = { 1, n_out, 1, 1 };
        ggml_tensor s = oracle_tensor(GGML_TYPE_F32, s_ne);
        ggml_tensor i = oracle_tensor(GGML_TYPE_I32, i_ne);
        ggml_tensor d = oracle_tensor(GGML_TYPE_F32, d_ne);
        emit(s, false, i, d, th, vt);
    }

    // Layouts: rows must be contiguous within themselves, not between each other.
    for (ggml_type st : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K }) {
        const int64_t w = (st == GGML_TYPE_Q4_K) ? 512 : 64;
        for (int skind : { 0, 1, 2, 3 }) for (int dkind : { 0, 1, 2, 3 }) for (int tiled = 0; tiled < 2; tiled++) {
            emit_shape(st, st == GGML_TYPE_F32 || st == GGML_TYPE_F16 ? st : GGML_TYPE_F32, GGML_TYPE_I32, w, idx_shapes[2],
                       tiled, 6, 8u << 20, skind, dkind);
        }
    }
    // an index tensor that is not row-contiguous
    {
        const int64_t s_ne[4] = { 64, 100, 2, 3 };
        const int64_t i_ne[4] = { 8, 2, 1, 1 };
        const int64_t d_ne[4] = { 64, 8, 2, 1 };
        ggml_tensor s = oracle_tensor(GGML_TYPE_F32, s_ne);
        ggml_tensor d = oracle_tensor(GGML_TYPE_F32, d_ne);
        const size_t strided[4] = { 8, 64, 128, 128 };
        ggml_tensor i = oracle_tensor_nb(GGML_TYPE_I32, i_ne, strided);
        emit(s, false, i, d, 6, 8u << 20);
        const size_t padded[4] = { 4, 48, 96, 96 };
        ggml_tensor ip = oracle_tensor_nb(GGML_TYPE_I32, i_ne, padded);
        emit(s, false, ip, d, 6, 8u << 20);
    }

    // Refusals: types the DSP does not gather, index types, and table/destination pairs.
    for (ggml_type st : { GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4,
                          GGML_TYPE_I64, GGML_TYPE_F64, GGML_TYPE_I8, GGML_TYPE_I16 }) {
        const int64_t w = 256;
        for (ggml_type dt : { GGML_TYPE_F32, st }) for (int tiled = 0; tiled < 2; tiled++) {
            emit_shape(st, dt, GGML_TYPE_I32, w, idx_shapes[2], tiled, 6, 8u << 20);
        }
    }
    for (ggml_type it : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I16, GGML_TYPE_I8, GGML_TYPE_Q8_0 }) {
        emit_shape(GGML_TYPE_F32, GGML_TYPE_F32, it, 64, idx_shapes[2], false, 6, 8u << 20);
    }
    // same-type copies only for F32, I32 and F16; any other type must widen to F32
    for (ggml_type st : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I32, GGML_TYPE_BF16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,
                          GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q1_0, GGML_TYPE_Q2_0 })
    for (ggml_type dt : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I32, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,
                          GGML_TYPE_I64, GGML_TYPE_F64 })
    for (int tiled = 0; tiled < 2; tiled++) {
        emit_shape(st, dt, GGML_TYPE_I32, 256, idx_shapes[2], tiled, 6, 8u << 20);
    }

    fprintf(stderr, "%d cases\n", n_cases);
    return 0;
}
