// Case table for ARGSORT and TOP_K through the real `htp_kparams_solve`. Line format:
//   <op: 0 ARGSORT, 1 TOP_K> <sort order> <src0> | <dst> | <session threads> <session vtcm bytes>
//   => <12 words | unsupported>
// The words are `htp_sort_kernel_params`: n_threads, order, is_top_k, chunk_elems, n_chunks,
// vtcm_size, merge_elems, n_slots, then div_ne01 and div_ne01_ne02 {mp, l}. A TOP_K ignores the
// order word of its op_params.
#include "common.h"

namespace {

int n_cases = 0;

void emit(bool top_k, int order, const ggml_tensor & s, const ggml_tensor & d, uint32_t th, size_t vt) {
    ggml_hexagon_session sess;
    sess.n_threads = th;
    sess.vtcm_size = vt;

    ggml_tensor op = d;
    op.op     = top_k ? GGML_OP_TOP_K : GGML_OP_ARGSORT;
    op.src[0] = const_cast<ggml_tensor *>(&s);
    op.op_params[0] = order;

    htp_sort_kernel_params k;
    memset(&k, 0xAA, sizeof(k));
    const bool ok = htp_kparams_solve(&sess, &op, &k);

    printf("%d %d ", (int) top_k, order);
    oracle_print_tensor(s); printf(" | "); oracle_print_tensor(d);
    printf(" | %u %zu =>", th, vt);
    oracle_print_words(ok, &k, sizeof(k), sizeof(k) / 4);
    n_cases++;
}

// An F32 source [ne0, rows...] and the I32 result: [ne0, ...] for an argsort, [k, ...] for a top-k.
void emit_shape(bool top_k, int order, int64_t ne0, int64_t k, int64_t ne1, int64_t ne2, int64_t ne3, uint32_t th,
                size_t vt) {
    const int64_t s_ne[4] = { ne0, ne1, ne2, ne3 };
    const int64_t d_ne[4] = { top_k ? k : ne0, ne1, ne2, ne3 };
    ggml_tensor s = oracle_tensor(GGML_TYPE_F32, s_ne);
    ggml_tensor d = oracle_tensor(GGML_TYPE_I32, d_ne);
    emit(top_k, order, s, d, th, vt);
}

}  // namespace

int main() {
    const uint32_t threads[] = { 0, 1, 2, 4, 6, 8, 10 };
    const size_t   vtcms[]   = { 8u << 20, 1u << 20, 256u << 10, 64u << 10, 16u << 10, 4u << 10, 0 };

    // MoE routing as cera does it: one row of expert scores, descending.
    for (int64_t ne0 : { 8, 16, 32, 64, 128, 256 }) for (uint32_t th : threads) for (size_t vt : vtcms) {
        emit_shape(false, GGML_SORT_ORDER_DESC, ne0, ne0, 1, 1, 1, th, vt);
    }

    // One long row: the chunk counts and the merge buffers, over every width around a power of two.
    for (int64_t ne0 : { 1, 2, 3, 31, 32, 33, 100, 512, 1000, 1023, 1024, 1025, 1500, 2047, 2048, 2049, 4096, 5000, 8192,
                         16384, 32000, 32768, 65535, 65536, 65537, 100000, 128256, 151936, 262144, 524288 })
    for (uint32_t th : { 0u, 1u, 4u, 8u, 10u }) for (size_t vt : { (size_t) (8u << 20), (size_t) (1u << 20), (size_t) (64u << 10), (size_t) (16u << 10), (size_t) 0 }) {
        for (int order : { GGML_SORT_ORDER_ASC, GGML_SORT_ORDER_DESC }) {
            emit_shape(false, order, ne0, ne0, 1, 1, 1, th, vt);
        }
        for (int64_t k : { 1, 2, 8, 20, 64, 1024 }) {
            if (k > ne0) continue;
            emit_shape(true, GGML_SORT_ORDER_DESC, ne0, k, 1, 1, 1, th, vt);
        }
        emit_shape(true, GGML_SORT_ORDER_ASC, ne0, ne0 < 8 ? ne0 : 8, 1, 1, 1, th, vt);
    }

    // Several rows: thread caps, slot counts and the per-thread scratch against the budget.
    for (int64_t ne0 : { 1, 8, 32, 33, 64, 100, 128, 256, 1000, 1024, 4096, 16384, 65536, 131072 })
    for (int64_t ne1 : { 2, 7, 64 }) for (int64_t ne2 : { 1, 3 }) for (int64_t ne3 : { 1, 2 })
    for (bool top_k : { false, true }) for (uint32_t th : { 0u, 1u, 6u, 10u })
    for (size_t vt : { (size_t) (8u << 20), (size_t) (1u << 20), (size_t) (64u << 10), (size_t) 0 }) {
        if (ne0 * ne1 * ne2 * ne3 > 30 * 1000 * 1000) continue;
        const int64_t k = ne0 < 8 ? ne0 : 8;
        emit_shape(top_k, GGML_SORT_ORDER_DESC, ne0, k, ne1, ne2, ne3, th, vt);
    }

    // Orders other than 0 and 1 reach the kernel as given.
    for (int order : { 2, -1, 255 }) {
        emit_shape(false, order, 128, 128, 1, 1, 1, 6, 8u << 20);
    }

    // Refusals: a non-F32 source or a non-I32 destination.
    for (ggml_type st : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I32, GGML_TYPE_BF16 })
    for (ggml_type dt : { GGML_TYPE_I32, GGML_TYPE_I64, GGML_TYPE_F32, GGML_TYPE_F16 })
    for (bool top_k : { false, true }) {
        const int64_t ne[4] = { 128, 4, 1, 1 };
        ggml_tensor s = oracle_tensor(st, ne);
        ggml_tensor d = oracle_tensor(dt, ne);
        emit(top_k, GGML_SORT_ORDER_DESC, s, d, 6, 8u << 20);
    }
    // strided operands go through (the DSP reads through the row strides)
    for (int kind : { 1, 2, 3 }) for (bool top_k : { false, true }) {
        const int64_t ne[4] = { 128, 4, 2, 1 };
        ggml_tensor s = oracle_tensor(GGML_TYPE_F32, ne, kind);
        ggml_tensor d = oracle_tensor(GGML_TYPE_I32, ne, kind);
        emit(top_k, GGML_SORT_ORDER_DESC, s, d, 6, 8u << 20);
    }

    fprintf(stderr, "%d cases\n", n_cases);
    return 0;
}
