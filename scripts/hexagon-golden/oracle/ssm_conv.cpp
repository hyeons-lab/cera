// Case table for SSM_CONV through the real `htp_kparams_solve`. Line format:
//   <src0 window> | <src1 taps> | <dst> | <session threads> <session vtcm bytes> => <words | unsupported>
// The words are the 10 of `htp_ssm_conv_kernel_params`.
#include "common.h"

namespace {

int n_cases = 0;

// A dense F32 tensor with the given extents; `row_pad` bytes are added to every row stride.
ggml_tensor f32(const int64_t * ne, size_t row_pad = 0) {
    const size_t nb[4] = { 4, ne[0] * 4 + row_pad, (ne[0] * 4 + row_pad) * ne[1],
                           (ne[0] * 4 + row_pad) * ne[1] * ne[2] };
    return oracle_tensor_nb(GGML_TYPE_F32, ne, nb);
}

void emit_tensors(const ggml_tensor & s0, const ggml_tensor & s1, const ggml_tensor & d, uint32_t th, size_t vt) {
    ggml_hexagon_session sess;
    sess.n_threads = th;
    sess.vtcm_size = vt;

    ggml_tensor op = d;
    op.op     = GGML_OP_SSM_CONV;
    op.src[0] = const_cast<ggml_tensor *>(&s0);
    op.src[1] = const_cast<ggml_tensor *>(&s1);
    htp_ssm_conv_kernel_params k;
    memset(&k, 0xAA, sizeof(k));  // poison, so a field the solver forgets to set shows up
    const bool ok = htp_kparams_solve(&sess, &op, &k);

    oracle_print_tensor(s0); printf(" | "); oracle_print_tensor(s1); printf(" | "); oracle_print_tensor(d);
    printf(" | %u %zu =>", th, vt);
    oracle_print_words(ok, &k, sizeof(k), sizeof(k) / 4);
    n_cases++;
}

// The window, taps and output of a depthwise conv over `n_t` new positions of `d_inner` channels.
void emit(int64_t d_conv, int64_t d_inner, int64_t n_t, int64_t n_s, uint32_t th, size_t vt) {
    const int64_t s0_ne[4] = { d_conv - 1 + n_t, d_inner, n_s, 1 };
    const int64_t s1_ne[4] = { d_conv, d_inner, 1, 1 };
    const int64_t d_ne[4]  = { d_inner, n_t, n_s, 1 };
    emit_tensors(f32(s0_ne), f32(s1_ne), f32(d_ne), th, vt);
}

}  // namespace

int main() {
    const uint32_t threads[] = { 1, 2, 4, 6, 8 };
    const size_t   vtcms[]   = { 8u << 20, 1u << 20, 64u << 10, 0 };

    // The shapes LFM2 (d_conv 3) and the conformer convolution module send, at every session
    // thread count and VTCM budget: the single-position decode kernel and the tiled prefill one.
    for (int64_t dc : { 3, 4 }) for (int64_t di : { 1, 31, 32, 33, 96, 1024, 2048, 8192 })
    for (int64_t nt : { 1, 2, 16, 32, 1024 }) for (int64_t ns : { 1, 2 })
    for (uint32_t th : threads) for (size_t vt : vtcms) {
        emit(dc, di, nt, ns, th, vt);
    }

    // Other tap counts (the kernel takes up to 32), odd chunk lengths and budgets that bound the
    // channel tile, plus empty position and sequence counts.
    const size_t more_vtcms[] = { 8u << 20, 512u << 10, 256u << 10, 128u << 10, 64u << 10, 0 };
    for (int64_t dc : { 1, 2, 5, 9, 17, 31, 32 }) for (int64_t di : { 1, 33, 96, 1024, 4096 })
    for (int64_t nt : { 1, 3, 32, 129, 700 }) for (uint32_t th : { 1u, 4u, 8u }) for (size_t vt : more_vtcms) {
        emit(dc, di, nt, 1, th, vt);
    }
    for (int64_t dc : { 3, 9 }) for (int64_t di : { 64, 1024 }) for (int64_t nt : { 0, 1, 40 }) for (int64_t ns : { 0, 1, 3 })
    for (uint32_t th : { 4u, 8u }) for (size_t vt : { size_t(8u << 20), size_t(128u << 10) }) {
        emit(dc, di, nt, ns, th, vt);
    }
    for (int64_t di : { 100, 1000, 10000, 100000 }) for (int64_t nt : { 1, 8, 64 })
    for (uint32_t th : { 1u, 6u, 8u }) for (size_t vt : { size_t(8u << 20), size_t(2u << 20), size_t(0) }) {
        emit(3, di, nt, 1, th, vt);
    }

    // Operands the kernel refuses.
    {
        const int64_t s0_ne[4] = { 34, 1024, 1, 1 }, s1_ne[4] = { 3, 1024, 1, 1 }, d_ne[4] = { 1024, 32, 1, 1 };
        const ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I32, GGML_TYPE_BF16 };
        for (ggml_type t0 : types) for (ggml_type t1 : types) for (ggml_type td : types) {
            ggml_tensor s0 = oracle_tensor(t0, s0_ne), s1 = oracle_tensor(t1, s1_ne), d = oracle_tensor(td, d_ne);
            emit_tensors(s0, s1, d, 4, 8u << 20);
        }
        // d_conv at the limits
        emit(0, 1024, 32, 1, 4, 8u << 20);
        emit(32, 1024, 32, 1, 4, 8u << 20);
        emit(33, 1024, 32, 1, 4, 8u << 20);
        emit(64, 1024, 32, 1, 4, 8u << 20);
        emit(3, 0, 32, 1, 4, 8u << 20);
        // the fourth dimension, and src1 batched
        for (int which = 0; which < 4; which++) {
            int64_t a[4] = { 34, 1024, 1, 1 }, b[4] = { 3, 1024, 1, 1 }, c[4] = { 1024, 32, 1, 1 };
            int64_t * t = which == 0 ? a : which == 1 ? b : c;
            t[3] = 2;
            if (which == 3) { b[2] = 2; }
            emit_tensors(f32(a), f32(b), f32(c), 4, 8u << 20);
        }
        { int64_t a[4] = { 34, 1024, 1, 1 }, b[4] = { 3, 1024, 2, 1 }, c[4] = { 1024, 32, 1, 1 };
          emit_tensors(f32(a), f32(b), f32(c), 4, 8u << 20); }
        // extents that disagree with each other
        struct Case { int64_t a[4], b[4], c[4]; };
        const Case bad[] = {
            { { 35, 1024, 1, 1 }, { 3, 1024, 1, 1 }, { 1024, 32, 1, 1 } },  // window length
            { { 34, 1023, 1, 1 }, { 3, 1024, 1, 1 }, { 1024, 32, 1, 1 } },  // window channels
            { { 34, 1024, 2, 1 }, { 3, 1024, 1, 1 }, { 1024, 32, 1, 1 } },  // window sequences
            { { 34, 1024, 1, 1 }, { 3, 1023, 1, 1 }, { 1024, 32, 1, 1 } },  // tap channels
            { { 34, 1024, 1, 1 }, { 4, 1024, 1, 1 }, { 1024, 32, 1, 1 } },  // tap count
            { { 34, 1024, 1, 1 }, { 3, 1024, 1, 1 }, { 1023, 32, 1, 1 } },  // dst channels
            { { 34, 1024, 1, 1 }, { 3, 1024, 1, 1 }, { 1024, 31, 1, 1 } },  // dst positions
            { { 34, 1024, 1, 1 }, { 3, 1024, 1, 1 }, { 1024, 32, 2, 1 } },  // dst sequences
        };
        for (const Case & c : bad) emit_tensors(f32(c.a), f32(c.b), f32(c.c), 4, 8u << 20);
        // strides: a transposed or padded operand the DMA cannot read as rows
        const int64_t a[4] = { 34, 1024, 1, 1 }, b[4] = { 3, 1024, 1, 1 }, c[4] = { 1024, 32, 2, 1 };
        const int64_t a2[4] = { 34, 1024, 2, 1 };
        emit_tensors(f32(a, 16), f32(b), f32(c), 4, 8u << 20);   // padded window rows
        emit_tensors(f32(a), f32(b, 16), f32(c), 4, 8u << 20);   // padded tap rows
        emit_tensors(f32(a2, 0), f32(b), f32(c, 16), 4, 8u << 20);   // padded dst rows: accepted
        emit_tensors(oracle_tensor(GGML_TYPE_F32, a, 2), f32(b), f32(c), 4, 8u << 20);  // transposed window
        emit_tensors(f32(a), oracle_tensor(GGML_TYPE_F32, b, 2), f32(c), 4, 8u << 20);  // transposed taps
        emit_tensors(f32(a), f32(b), oracle_tensor(GGML_TYPE_F32, c, 2), 4, 8u << 20);  // transposed dst
        {   // the sequence stride is the kernel's own business, whatever it is
            ggml_tensor s0 = f32(a2), s1 = f32(b), d = f32(c);
            s0.nb[2] = s0.nb[2] * 3 + 128;
            d.nb[2]  = d.nb[2] + 64;
            emit_tensors(s0, s1, d, 4, 8u << 20);
        }
    }

    fprintf(stderr, "%d cases\n", n_cases);
    return 0;
}
