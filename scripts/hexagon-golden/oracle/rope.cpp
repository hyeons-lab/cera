// Case table for ROPE through the real `htp_kparams_solve`. Line format:
//   <src0> | <positions> | <freq factors or -> | <dst> | <16 op_params words> | <session threads> <session vtcm bytes>
//   => <6 words | unsupported>
// The words are `htp_rope_kernel_params`: n_threads, vtcm_size, div_ne2_ne1 {mp, l}, div_ne1 {mp, l}.
#include "common.h"

#include <vector>

namespace {

struct rope_args {
    int     n_dims     = 128;
    int     mode       = GGML_ROPE_TYPE_NEOX;
    int     truncate   = 0;  // op_params[3]; the DSP rope only does the truncating form
    int     n_offs     = 0;
    float   freq_base  = 10000.0f;
    int32_t sections[4] = { 0, 0, 0, 0 };
};

void fill_params(int32_t * p, const rope_args & a) {
    memset(p, 0, 16 * sizeof(int32_t));
    p[1] = a.n_dims;
    p[2] = a.mode;
    p[3] = a.truncate;
    p[4] = 4096;  // n_ctx_orig
    const float words[6] = { a.freq_base, 1.0f /* freq_scale */, 0.0f /* ext_factor */, 1.0f /* attn_factor */,
                             32.0f /* beta_fast */, 1.0f /* beta_slow */ };
    memcpy(p + 5, words, sizeof(words));
    memcpy(p + 11, a.sections, sizeof(a.sections));
    p[15] = a.n_offs;
}

int n_cases = 0;

void emit(const ggml_tensor & s, const ggml_tensor & pos, const ggml_tensor * ff, const ggml_tensor & d,
          const rope_args & a, uint32_t th, size_t vt) {
    ggml_hexagon_session sess;
    sess.n_threads = th;
    sess.vtcm_size = vt;

    ggml_tensor op = d;
    op.op     = GGML_OP_ROPE;
    op.src[0] = const_cast<ggml_tensor *>(&s);
    op.src[1] = const_cast<ggml_tensor *>(&pos);
    op.src[2] = const_cast<ggml_tensor *>(ff);
    int32_t p[16];
    fill_params(p, a);
    memcpy(op.op_params, p, sizeof(p));

    htp_rope_kernel_params k;
    memset(&k, 0xAA, sizeof(k));  // poison, so a field the solver forgets to set shows up
    const bool ok = htp_kparams_solve(&sess, &op, &k);

    oracle_print_tensor(s); printf(" | "); oracle_print_tensor(pos); printf(" | ");
    if (ff) { oracle_print_tensor(*ff); } else { printf("-"); }
    printf(" | "); oracle_print_tensor(d); printf(" |");
    for (int i = 0; i < 16; i++) printf(" %u", (uint32_t) p[i]);
    printf(" | %u %zu =>", th, vt);
    oracle_print_words(ok, &k, sizeof(k), sizeof(k) / 4);
    n_cases++;
}

// A contiguous F32 activation [ne00, nh, nt, nb], rotated in place (dst has the same layout), with
// `npos` positions and optional `nff` frequency factors.
void emit_basic(int64_t ne00, int64_t nh, int64_t nt, int64_t nb, int kind, const rope_args & a, int64_t npos,
                int64_t nff, uint32_t th, size_t vt) {
    const int64_t s_ne[4] = { ne00, nh, nt, nb };
    const int64_t p_ne[4] = { npos, 1, 1, 1 };
    const int64_t f_ne[4] = { nff, 1, 1, 1 };
    ggml_tensor s = oracle_tensor(GGML_TYPE_F32, s_ne, kind);
    ggml_tensor p = oracle_tensor(GGML_TYPE_I32, p_ne);
    ggml_tensor f = oracle_tensor(GGML_TYPE_F32, f_ne);
    emit(s, p, nff > 0 ? &f : nullptr, s, a, th, vt);
}

}  // namespace

int main() {
    const uint32_t threads[] = { 0, 1, 4, 6, 8 };
    const size_t   vtcms[]   = { 8u << 20, 1u << 20, 64u << 10, 0 };

    // The shapes cera sends (decode: one token; prefill: a chunk), the full thread and budget grid.
    for (int64_t ne00 : { 64, 128, 256 }) for (int64_t nh : { 8, 32 }) for (int64_t nt : { 1, 64 })
    for (int mode : { GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX }) for (uint32_t th : threads) for (size_t vt : vtcms) {
        rope_args a;
        a.n_dims = (int) ne00;
        a.mode   = mode;
        emit_basic(ne00, nh, nt, 1, 0, a, nt, 0, th, vt);
    }

    // Row widths, head counts, token counts, batches and layouts, at two session settings.
    for (int64_t ne00 : { 32, 64, 80, 96, 128, 192, 256, 512, 1024, 4096, 16384 })
    for (int64_t nh : { 1, 2, 3, 8, 16, 32 }) for (int64_t nt : { 1, 2, 7, 64, 513 }) for (int64_t nb : { 1, 2 })
    for (int kind : { 0, 1, 3 }) {
        rope_args a;
        a.n_dims = (int) ne00;
        emit_basic(ne00, nh, nt, nb, kind, a, nt, 0, 6, 8u << 20);
        emit_basic(ne00, nh, nt, nb, kind, a, nt, 0, 1, 64u << 10);
    }

    // The same through every supported mode, including the multi-axis and vision forms.
    for (int mode : { GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX, GGML_ROPE_TYPE_MROPE, GGML_ROPE_TYPE_VISION,
                      GGML_ROPE_TYPE_IMROPE, 1, 3, 4, 6, 16, 32, 64 })
    for (int64_t ne00 : { 64, 128 }) for (int64_t nt : { 1, 9 }) {
        rope_args a;
        a.n_dims      = (mode == GGML_ROPE_TYPE_VISION) ? (int) (ne00 / 2) : (int) ne00;
        a.mode        = mode;
        a.sections[0] = 16; a.sections[1] = 24; a.sections[2] = 24; a.sections[3] = 0;
        const bool multi = (mode & GGML_ROPE_TYPE_MROPE) != 0;
        emit_basic(ne00, 8, nt, 1, 0, a, multi ? nt * 4 : nt, 0, 6, 8u << 20);
        emit_basic(ne00, 8, nt, 1, 0, a, multi ? nt * 4 - 1 : nt - 1, 0, 6, 8u << 20);  // one position short
        // a mode that needs sections but has none
        rope_args none = a;
        memset(none.sections, 0, sizeof(none.sections));
        emit_basic(ne00, 8, nt, 1, 0, none, multi ? nt * 4 : nt, 0, 6, 8u << 20);
        rope_args neg = a;
        neg.sections[0] = -1; neg.sections[1] = 0; neg.sections[2] = 8;
        emit_basic(ne00, 8, nt, 1, 0, neg, multi ? nt * 4 : nt, 0, 6, 8u << 20);
        // the solver looks at the first three section words only: a positive fourth does not count,
        // a positive third does
        rope_args fourth = none;
        fourth.sections[3] = 8;
        emit_basic(ne00, 8, nt, 1, 0, fourth, multi ? nt * 4 : nt, 0, 6, 8u << 20);
        rope_args third = none;
        third.sections[2] = 8;
        emit_basic(ne00, 8, nt, 1, 0, third, multi ? nt * 4 : nt, 0, 6, 8u << 20);
    }
    // vision rotates the full row: any other n_dims or an offset is refused
    for (int n_dims : { 16, 32, 48, 64 }) for (int n_offs : { 0, 32 }) {
        rope_args a;
        a.n_dims = n_dims;
        a.mode   = GGML_ROPE_TYPE_VISION;
        a.n_offs = n_offs;
        a.sections[0] = a.sections[1] = a.sections[2] = 8;
        emit_basic(64, 8, 4, 1, 0, a, 16, 0, 6, 8u << 20);
    }

    // Partial rotary: n_dims below the row width, with and without an offset.
    for (int64_t ne00 : { 64, 128, 256 }) for (int mode : { GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX })
    for (int n_dims : { 0, 2, 16, 30, 31, 32, 48, 64, 96, 128, 200, 256, 300 })
    for (int n_offs : { 0, 16, 32, 64, 96, 128, 224, -32 }) {
        rope_args a;
        a.n_dims = n_dims;
        a.mode   = mode;
        a.n_offs = n_offs;
        emit_basic(ne00, 4, 3, 1, 0, a, 3, 0, 6, 8u << 20);
    }

    // Frequency factors (the lfm3 global layers): enough, exactly enough, too few, none, and a
    // buffer large enough to change the VTCM total.
    for (int64_t ne00 : { 64, 128, 256 }) for (int n_dims : { (int) ne00, (int) ne00 / 2 })
    for (int64_t nff : { 0, 1, 15, 16, 31, 32, 33, 63, 64, 65, 100, 128, 129, 1000 }) for (uint32_t th : { 1u, 6u, 8u })
    for (size_t vt : { (size_t) (8u << 20), (size_t) (1u << 20), (size_t) 0 }) {
        rope_args a;
        a.n_dims = n_dims;
        emit_basic(ne00, 8, 5, 1, 0, a, 5, nff, th, vt);
    }

    // Operand mismatches the solver refuses.
    {
        rope_args a;
        a.n_dims = 128;
        const int64_t ne[4] = { 128, 8, 4, 1 };
        const int64_t pne[4] = { 4, 1, 1, 1 };
        ggml_tensor s = oracle_tensor(GGML_TYPE_F32, ne);
        ggml_tensor p = oracle_tensor(GGML_TYPE_I32, pne);

        // non-F32 source or destination, F32 positions, I64 positions
        ggml_tensor s16 = oracle_tensor(GGML_TYPE_F16, ne);
        emit(s16, p, nullptr, s16, a, 6, 8u << 20);
        emit(s, p, nullptr, s16, a, 6, 8u << 20);
        emit(s16, p, nullptr, s, a, 6, 8u << 20);
        ggml_tensor pf = oracle_tensor(GGML_TYPE_F32, pne);
        emit(s, pf, nullptr, s, a, 6, 8u << 20);
        ggml_tensor p64 = oracle_tensor(GGML_TYPE_I64, pne);
        emit(s, p64, nullptr, s, a, 6, 8u << 20);

        // destination of another shape
        const int64_t dne[4] = { 128, 8, 5, 1 };
        ggml_tensor d = oracle_tensor(GGML_TYPE_F32, dne);
        emit(s, p, nullptr, d, a, 6, 8u << 20);

        // positions that are not contiguous (a strided view)
        const size_t pnb[4] = { 4, 16, 64, 64 };
        const int64_t pne2[4] = { 2, 2, 1, 1 };
        ggml_tensor ps = oracle_tensor_nb(GGML_TYPE_I32, pne2, pnb);
        emit(s, ps, nullptr, s, a, 6, 8u << 20);
        const size_t pnb_t[4] = { 8, 32, 32, 32 };
        const int64_t pne3[4] = { 4, 1, 1, 1 };
        ggml_tensor pst = oracle_tensor_nb(GGML_TYPE_I32, pne3, pnb_t);
        emit(s, pst, nullptr, s, a, 6, 8u << 20);

        // destination with a strided dim 0 or a row stride below the row width
        const size_t bad0[4] = { 8, 1024, 8192, 32768 };
        ggml_tensor d0 = oracle_tensor_nb(GGML_TYPE_F32, ne, bad0);
        emit(s, p, nullptr, d0, a, 6, 8u << 20);
        emit(d0, p, nullptr, s, a, 6, 8u << 20);
        const size_t short1[4] = { 4, 256, 2048, 8192 };
        ggml_tensor d1 = oracle_tensor_nb(GGML_TYPE_F32, ne, short1);
        emit(s, p, nullptr, d1, a, 6, 8u << 20);
        emit(d1, p, nullptr, s, a, 6, 8u << 20);

        // a transposed source (dim 0 strided)
        ggml_tensor st = oracle_tensor(GGML_TYPE_F32, ne, 2);
        emit(st, p, nullptr, s, a, 6, 8u << 20);

        // the non-truncating form, negative base, odd and negative n_dims
        rope_args t = a; t.truncate = 1;
        emit(s, p, nullptr, s, t, 6, 8u << 20);
        rope_args nb = a; nb.freq_base = -1.0f;
        emit(s, p, nullptr, s, nb, 6, 8u << 20);
        rope_args zb = a; zb.freq_base = 0.0f;
        emit(s, p, nullptr, s, zb, 6, 8u << 20);
        rope_args od = a; od.n_dims = 127;
        emit(s, p, nullptr, s, od, 6, 8u << 20);
        rope_args ng = a; ng.n_dims = -2;
        emit(s, p, nullptr, s, ng, 6, 8u << 20);

        // frequency factors of the wrong type or not contiguous
        const int64_t fne[4] = { 64, 1, 1, 1 };
        ggml_tensor f16 = oracle_tensor(GGML_TYPE_F16, fne);
        emit(s, p, &f16, s, a, 6, 8u << 20);
        const size_t fnb[4] = { 8, 512, 512, 512 };
        ggml_tensor fs = oracle_tensor_nb(GGML_TYPE_F32, fne, fnb);
        emit(s, p, &fs, s, a, 6, 8u << 20);

        // empty tensors
        const int64_t zne[4] = { 128, 0, 4, 1 };
        ggml_tensor z = oracle_tensor(GGML_TYPE_F32, zne);
        emit(z, p, nullptr, z, a, 6, 8u << 20);
        const int64_t zne0[4] = { 0, 8, 4, 1 };
        ggml_tensor z0 = oracle_tensor(GGML_TYPE_F32, zne0);
        emit(z0, p, nullptr, z0, a, 6, 8u << 20);
    }

    fprintf(stderr, "%d cases\n", n_cases);
    return 0;
}
