// Case table for SOFT_MAX through the real `htp_kparams_solve`. Line format:
//   <src0> | <mask or -> | <dst> | <scale bits> <max_bias bits> <has sinks> | <session threads> <session vtcm bytes>
//   => <19 words | unsupported>
// The words are `htp_softmax_kernel_params`: kernel_type, n_threads, vtcm_size, n_head, n_head_log2,
// use_src1, use_f16, scale, max_bias, m0, m1, then div_ne01, div_ne02, div_ne12, div_ne13 {mp, l}.
#include "common.h"

namespace {

int n_cases = 0;

uint32_t bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

void emit(const ggml_tensor & s, const ggml_tensor * m, const ggml_tensor & d, float scale, float max_bias,
          bool sinks, uint32_t th, size_t vt) {
    ggml_hexagon_session sess;
    sess.n_threads = th;
    sess.vtcm_size = vt;

    const int64_t sink_ne[4] = { s.ne[2], 1, 1, 1 };
    ggml_tensor sink = oracle_tensor(GGML_TYPE_F32, sink_ne);

    ggml_tensor op = d;
    op.op     = GGML_OP_SOFT_MAX;
    op.src[0] = const_cast<ggml_tensor *>(&s);
    op.src[1] = const_cast<ggml_tensor *>(m);
    op.src[2] = sinks ? &sink : nullptr;
    memcpy(&op.op_params[0], &scale, sizeof(float));
    memcpy(&op.op_params[1], &max_bias, sizeof(float));

    htp_softmax_kernel_params k;
    memset(&k, 0xAA, sizeof(k));
    const bool ok = htp_kparams_solve(&sess, &op, &k);

    oracle_print_tensor(s); printf(" | ");
    if (m) { oracle_print_tensor(*m); } else { printf("-"); }
    printf(" | "); oracle_print_tensor(d);
    printf(" | %u %u %d | %u %zu =>", bits(scale), bits(max_bias), (int) sinks, th, vt);
    oracle_print_words(ok, &k, sizeof(k), sizeof(k) / 4);
    n_cases++;
}

// Mask of `ne` extents and `mtype`, contiguous unless `kind` says otherwise.
ggml_tensor mask_tensor(ggml_type mtype, int64_t m0, int64_t m1, int64_t m2, int64_t m3, int kind = 0) {
    const int64_t ne[4] = { m0, m1, m2, m3 };
    return oracle_tensor(mtype, ne, kind);
}

}  // namespace

int main() {
    const uint32_t threads[] = { 0, 1, 4, 6, 8 };
    const size_t   vtcms[]   = { 8u << 20, 1u << 20, 64u << 10, 0 };

    // The attention score shapes cera sends: the full thread and budget grid.
    for (int64_t ne00 : { 126, 512, 1500 }) for (int64_t ne01 : { 1, 126 }) for (int64_t heads : { 1, 8, 12 })
    for (int mask : { 0, 1, 2 }) for (uint32_t th : threads) for (size_t vt : vtcms) {
        const int64_t ne[4] = { ne00, ne01, heads, 1 };
        ggml_tensor s = oracle_tensor(GGML_TYPE_F32, ne);
        ggml_tensor m = mask_tensor(mask == 2 ? GGML_TYPE_F16 : GGML_TYPE_F32, ne00, ne01, heads, 1);
        emit(s, mask ? &m : nullptr, s, 0.125f, 0.0f, false, th, vt);
    }

    // Row widths, row counts, head counts and batches at two session settings.
    for (int64_t ne00 : { 1, 7, 32, 33, 128, 1000, 4096, 16384, 32768 })
    for (int64_t ne01 : { 1, 3, 126 }) for (int64_t heads : { 1, 3, 8, 12, 32, 512 })
    for (int64_t ne03 : { 1, 2 }) for (int mask : { 0, 1, 2 }) for (int kind : { 0, 1 }) {
        if (ne01 * heads * ne03 * ne00 > 40 * 1000 * 1000) continue;
        const int64_t ne[4] = { ne00, ne01, heads, ne03 };
        ggml_tensor s = oracle_tensor(GGML_TYPE_F32, ne, kind);
        ggml_tensor d = oracle_tensor(GGML_TYPE_F32, ne, kind);
        ggml_tensor m = mask_tensor(mask == 2 ? GGML_TYPE_F16 : GGML_TYPE_F32, ne00, ne01, heads, ne03);
        emit(s, mask ? &m : nullptr, d, 1.0f / 8.0f, 0.0f, false, 6, 8u << 20);
        emit(s, mask ? &m : nullptr, d, 1.0f, 0.0f, false, 1, 64u << 10);
    }

    // Masks: a taller mask (a shifted view of a larger one), a shorter one, a broadcast over heads
    // and batches (including counts that do not divide), a wider or narrower row.
    for (int64_t ne02 : { 1, 8, 12 }) for (int64_t ne03 : { 1, 2 }) {
        const int64_t ne[4] = { 64, 8, ne02, ne03 };
        ggml_tensor s = oracle_tensor(GGML_TYPE_F32, ne);
        for (ggml_type mt : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
            for (int64_t m1 : { 7, 8, 64 }) for (int64_t m2 : { 1, 2, 3, 4, 8, 24 }) for (int64_t m3 : { 1, 2, 4 }) {
                ggml_tensor m = mask_tensor(mt, 64, m1, m2, m3);
                emit(s, &m, s, 0.5f, 0.0f, false, 6, 8u << 20);
            }
            for (int64_t m0 : { 32, 63, 65, 128 }) {
                ggml_tensor m = mask_tensor(mt, m0, 8, ne02, ne03);
                emit(s, &m, s, 0.5f, 0.0f, false, 6, 8u << 20);
            }
        }
    }

    // A mask that is a strided view (the audio encoder's relative-position shift is one): the host
    // solver wants a contiguous mask, whatever the strides between its rows.
    for (int64_t t : { 16, 125, 126 }) for (ggml_type mt : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
        const int64_t ne[4] = { t, t, 8, 1 };
        ggml_tensor s = oracle_tensor(GGML_TYPE_F32, ne);
        const size_t ts = ggml_type_size(mt);
        const int64_t lp = (2 * t - 1 + 31) / 32 * 32;
        const size_t nb[4] = { ts, (size_t) (lp - 1) * ts, (size_t) (t * lp) * ts, (size_t) (t * lp * 8) * ts };
        ggml_tensor m = oracle_tensor_nb(mt, ne, nb);
        emit(s, &m, s, 0.125f, 0.0f, false, 6, 8u << 20);
        for (int kind : { 1, 2, 3 }) {
            ggml_tensor mk = mask_tensor(mt, t, t, 8, 1, kind);
            emit(s, &mk, s, 0.125f, 0.0f, false, 6, 8u << 20);
        }
    }

    // ALiBi slopes: positive max_bias with head counts that are and are not powers of two.
    for (float max_bias : { 0.0f, 8.0f, 16.0f, 0.5f, 1.0f, -8.0f })
    for (int64_t heads : { 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 64, 100, 512 }) for (int mask : { 0, 1, 2 }) {
        const int64_t ne[4] = { 128, 4, heads, 1 };
        ggml_tensor s = oracle_tensor(GGML_TYPE_F32, ne);
        ggml_tensor m = mask_tensor(mask == 2 ? GGML_TYPE_F16 : GGML_TYPE_F32, 128, 4, heads, 1);
        emit(s, mask ? &m : nullptr, s, 0.25f, max_bias, false, 6, 8u << 20);
    }

    // Scales.
    for (float scale : { 1.0f, 0.0f, -1.0f, 0.125f, 0.088388347648f, 1e-30f, 3.0e10f }) {
        const int64_t ne[4] = { 128, 4, 8, 1 };
        ggml_tensor s = oracle_tensor(GGML_TYPE_F32, ne);
        emit(s, nullptr, s, scale, 0.0f, false, 6, 8u << 20);
    }

    // Refusals: sinks, non-F32 operands, non-contiguous source or destination, more than 512 heads.
    {
        const int64_t ne[4] = { 128, 4, 8, 1 };
        ggml_tensor s = oracle_tensor(GGML_TYPE_F32, ne);
        ggml_tensor m = mask_tensor(GGML_TYPE_F32, 128, 4, 8, 1);
        emit(s, nullptr, s, 1.0f, 0.0f, true, 6, 8u << 20);
        emit(s, &m, s, 1.0f, 0.0f, true, 6, 8u << 20);

        ggml_tensor s16 = oracle_tensor(GGML_TYPE_F16, ne);
        emit(s16, nullptr, s16, 1.0f, 0.0f, false, 6, 8u << 20);
        emit(s, nullptr, s16, 1.0f, 0.0f, false, 6, 8u << 20);
        emit(s16, nullptr, s, 1.0f, 0.0f, false, 6, 8u << 20);
        ggml_tensor mbf = mask_tensor(GGML_TYPE_BF16, 128, 4, 8, 1);
        emit(s, &mbf, s, 1.0f, 0.0f, false, 6, 8u << 20);
        ggml_tensor mi = mask_tensor(GGML_TYPE_I32, 128, 4, 8, 1);
        emit(s, &mi, s, 1.0f, 0.0f, false, 6, 8u << 20);

        ggml_tensor sp = oracle_tensor(GGML_TYPE_F32, ne, 1);
        emit(sp, nullptr, s, 1.0f, 0.0f, false, 6, 8u << 20);
        emit(s, nullptr, sp, 1.0f, 0.0f, false, 6, 8u << 20);
        ggml_tensor st = oracle_tensor(GGML_TYPE_F32, ne, 2);
        emit(st, nullptr, s, 1.0f, 0.0f, false, 6, 8u << 20);
        emit(s, nullptr, st, 1.0f, 0.0f, false, 6, 8u << 20);

        const int64_t ne513[4] = { 128, 4, 513, 1 };
        ggml_tensor s513 = oracle_tensor(GGML_TYPE_F32, ne513);
        emit(s513, nullptr, s513, 1.0f, 0.0f, false, 6, 8u << 20);
        const int64_t ne512[4] = { 128, 4, 512, 1 };
        ggml_tensor s512 = oracle_tensor(GGML_TYPE_F32, ne512);
        emit(s512, nullptr, s512, 1.0f, 0.0f, false, 6, 8u << 20);
    }

    // Empty row counts (no rows, nothing to compute).
    for (int64_t ne01 : { 0 }) for (int64_t ne03 : { 1, 0 }) {
        const int64_t ne[4] = { 128, ne01, 8, ne03 };
        ggml_tensor s = oracle_tensor(GGML_TYPE_F32, ne);
        emit(s, nullptr, s, 1.0f, 0.0f, false, 6, 8u << 20);
        emit(s, nullptr, s, 1.0f, 0.0f, false, 0, 8u << 20);
    }

    fprintf(stderr, "%d cases\n", n_cases);
    return 0;
}
