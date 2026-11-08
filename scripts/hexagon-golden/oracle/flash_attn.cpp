// Case table for FLASH_ATTN_EXT through the real `htp_kparams_solve`. The solver reads only the
// extents and types of its operands (never the strides), so a line carries `type ne0 ne1 ne2 ne3`
// per tensor. Line format:
//   <q> | <k> | <v> | <mask or -> | <sinks or -> | <dst> | <scale bits> <max_bias bits> <softcap bits> <kv rows>
//     | <threads> <vtcm bytes> <n_hmx> <mdev count> <opt_fa_select> <opt_fa_split> => <words | unsupported>
// The float op params are printed as their f32 bit patterns so the Rust side reproduces them exactly.
#include "common.h"

#include <initializer_list>

struct Case {
    int64_t   dk = 64, dv = 64, nh = 16, nkv = 8, m = 1, s = 512;
    int64_t   n3 = 1;           // q->ne[3]
    int64_t   k3 = 1, v3 = 1;   // k->ne[3], v->ne[3]
    int64_t   v2 = -1;          // v->ne[2], nkv when negative
    ggml_type qt = GGML_TYPE_F32, kt = GGML_TYPE_F16, vt = GGML_TYPE_F16, dt = GGML_TYPE_F32;
    int       mask = 1;         // 0 none, 1 [s, m, 1, n3], 2 [s, m, nh, n3], 3 [s, pad64(m), 1, n3], 4 [s, m, 1, 1]
    ggml_type mt = GGML_TYPE_F16;
    int       sinks = 0;        // 0 none, 1 present
    ggml_type st = GGML_TYPE_F32;
    int       rows = 0;         // src[5] (K/V row lists) set
    float     scale = 0.125f, max_bias = 0.0f, softcap = 0.0f;
    uint32_t  th = 6;
    size_t    vtcm = 8u << 20;
    uint32_t  hmx = 1, mdev = 1;
    int       select = 2, split = 1;
};

static int g_cases = 0;

static void print_t(const ggml_tensor & t) {
    printf("%d %lld %lld %lld %lld", (int) t.type, (long long) t.ne[0], (long long) t.ne[1], (long long) t.ne[2],
           (long long) t.ne[3]);
}

static uint32_t bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

extern int opt_fa_select;
extern int opt_fa_split;

static void emit(const Case & c) {
    const int64_t q_ne[4] = { c.dk, c.m, c.nh, c.n3 };
    const int64_t k_ne[4] = { c.dk, c.s, c.nkv, c.k3 };
    const int64_t v_ne[4] = { c.dv, c.s, c.v2 < 0 ? c.nkv : c.v2, c.v3 };
    const int64_t d_ne[4] = { c.dv, c.nh, c.m, c.n3 };
    int64_t       m_ne[4] = { c.s, c.m, 1, c.n3 };
    switch (c.mask) {
        case 2: m_ne[2] = c.nh; break;
        case 3: m_ne[1] = (c.m + 63) / 64 * 64; break;
        case 4: m_ne[3] = 1; break;
        default: break;
    }
    const int64_t s_ne[4] = { c.nh, 1, 1, 1 };
    const int64_t r_ne[4] = { 4, 1, 1, 1 };

    ggml_tensor q = oracle_tensor(c.qt, q_ne), k = oracle_tensor(c.kt, k_ne), v = oracle_tensor(c.vt, v_ne);
    ggml_tensor d = oracle_tensor(c.dt, d_ne), mk = oracle_tensor(c.mt, m_ne), sk = oracle_tensor(c.st, s_ne);
    ggml_tensor rows = oracle_tensor(GGML_TYPE_I32, r_ne);

    ggml_hexagon_session sess;
    sess.n_threads  = c.th;
    sess.vtcm_size  = c.vtcm;
    sess.n_hmx      = c.hmx;
    sess.mdev_count = c.mdev;
    opt_fa_select   = c.select;
    opt_fa_split    = c.split;

    ggml_tensor op = d;
    op.op     = GGML_OP_FLASH_ATTN_EXT;
    op.src[0] = &q;
    op.src[1] = &k;
    op.src[2] = &v;
    op.src[3] = c.mask ? &mk : nullptr;
    op.src[4] = c.sinks ? &sk : nullptr;
    op.src[5] = c.rows ? &rows : nullptr;
    memcpy(&op.op_params[0], &c.scale, sizeof(float));
    memcpy(&op.op_params[1], &c.max_bias, sizeof(float));
    memcpy(&op.op_params[2], &c.softcap, sizeof(float));

    htp_fa_kernel_params kp;
    memset(&kp, 0xAA, sizeof(kp));  // poison, so a field the solver forgets to set shows up
    const bool ok = htp_kparams_solve(&sess, &op, &kp);

    print_t(q); printf(" | "); print_t(k); printf(" | "); print_t(v); printf(" | ");
    if (c.mask) print_t(mk); else printf("-");
    printf(" | ");
    if (c.sinks) print_t(sk); else printf("-");
    printf(" | "); print_t(d);
    printf(" | %u %u %u %d", bits(c.scale), bits(c.max_bias), bits(c.softcap), c.rows);
    printf(" | %u %zu %u %u %d %d =>", c.th, c.vtcm, c.hmx, c.mdev, c.select, c.split);
    oracle_print_words(ok, &kp, sizeof(kp), (int) ((sizeof(kp) + 3) / 4));
    g_cases++;

    opt_fa_select = 2;
    opt_fa_split  = 1;
}

int main() {
    // Models cera runs: (head dim K, head dim V, heads, kv heads). LFM2/LFM3 (GQA with 8 kv heads at
    // head dims 64 and 128), the vision towers and Whisper (MHA), and head dims that are not a
    // multiple of 64 (HMX pads them) or of 8 (HVX only).
    struct Shape { int64_t dk, dv, nh, nkv; };
    const Shape shapes[] = {
        { 64, 64, 16, 8 },  { 64, 64, 32, 8 },   { 128, 128, 32, 8 }, { 128, 128, 64, 8 }, { 64, 64, 12, 12 },
        { 64, 64, 16, 16 }, { 72, 72, 16, 16 },  { 96, 96, 12, 12 },  { 128, 128, 8, 8 },  { 64, 64, 8, 1 },
        { 256, 256, 8, 4 }, { 80, 80, 20, 20 },  { 70, 70, 8, 2 },
    };
    const int64_t ms[] = { 1, 2, 4, 5, 8, 32, 64, 128, 512 };
    const int64_t ss[] = { 1, 63, 64, 192, 193, 512, 768, 1024, 1500, 2048, 4096, 8192 };

    // 1. Query-count x KV-length sweep per model shape, default session, mask on and off.
    for (const Shape & sh : shapes) for (int64_t m : ms) for (int64_t s : ss) for (int mask : { 1, 0 }) {
        if (mask == 0 && (m == 2 || m == 8)) continue;
        Case c;
        c.dk = sh.dk; c.dv = sh.dv; c.nh = sh.nh; c.nkv = sh.nkv; c.m = m; c.s = s; c.mask = mask;
        emit(c);
    }

    // 2. Session sweep: thread counts, VTCM budgets, with and without an HMX unit.
    const Shape rep[] = { { 64, 64, 16, 8 }, { 128, 128, 32, 8 }, { 64, 64, 12, 12 }, { 256, 256, 16, 4 } };
    const int64_t reps[][2] = { { 1, 2048 }, { 32, 512 }, { 256, 4096 }, { 1, 100 } };
    for (const Shape & sh : rep) for (const auto & ms_ : reps)
    for (uint32_t th : { 0u, 1u, 2u, 4u, 6u, 8u }) for (size_t vt : { (size_t) 8 << 20, (size_t) 1 << 20, (size_t) 256 << 10, (size_t) 64 << 10, (size_t) 0 })
    for (uint32_t hmx : { 0u, 1u }) {
        Case c;
        c.dk = sh.dk; c.dv = sh.dv; c.nh = sh.nh; c.nkv = sh.nkv; c.m = ms_[0]; c.s = ms_[1];
        c.th = th; c.vtcm = vt; c.hmx = hmx;
        emit(c);
    }
    // a few budgets around the HMX/HVX boundary so the fallback and the failure both show up
    for (size_t vt : { (size_t) 2 << 20, (size_t) 1536 << 10, (size_t) 768 << 10, (size_t) 512 << 10, (size_t) 384 << 10, (size_t) 128 << 10, (size_t) 96 << 10 })
    for (int64_t m : { 1, 16, 64, 512 }) for (int64_t s : { 256, 4096 }) for (uint32_t th : { 4u, 8u }) {
        Case c;
        c.m = m; c.s = s; c.th = th; c.vtcm = vt;
        emit(c);
    }

    // 3. Host options and multi-device teams: opt_fa_select, opt_fa_split, mdev count x kv heads.
    for (int sel : { 0, 1, 2, 3 }) for (int split : { 0, 1 }) for (uint32_t mdev : { 1u, 2u, 3u, 4u, 6u, 8u })
    for (int64_t nkv : { 1, 2, 8, 12 }) for (const auto & ms_ : { std::pair<int64_t, int64_t>{ 1, 2048 }, { 64, 1024 }, { 1, 100 } })
    for (uint32_t th : { 4u, 8u }) {
        Case c;
        c.nkv = nkv; c.nh = nkv * 4; c.m = ms_.first; c.s = ms_.second;
        c.th = th; c.mdev = mdev; c.select = sel; c.split = split;
        emit(c);
    }
    for (int sel : { 0, 1, 2, 3 }) for (int split : { 0, 1 }) for (const Shape & sh : shapes) for (int64_t m : { 1, 5, 32 }) {
        Case c;
        c.dk = sh.dk; c.dv = sh.dv; c.nh = sh.nh; c.nkv = sh.nkv; c.m = m; c.s = 700; c.select = sel; c.split = split;
        emit(c);
    }

    // 4. Operand types. One axis at a time off the baseline, plus the k x v and q x dst grids, on a
    //    long-chunk (HMX) and a decode (HVX) shape.
    const ggml_type kvts[]  = { GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_BF16 };
    const ggml_type qts[]   = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0 };
    const ggml_type dsts[]  = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_BF16 };
    const ggml_type mts[]   = { GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_BF16 };
    const ggml_type sts[]   = { GGML_TYPE_F32, GGML_TYPE_F16 };
    for (const auto & ms_ : { std::pair<int64_t, int64_t>{ 64, 1024 }, { 1, 2048 }, { 1, 100 }, { 5, 64 } })
    for (int hdim : { 64, 128 }) {
        auto base = [&]() { Case c; c.dk = c.dv = hdim; c.m = ms_.first; c.s = ms_.second; return c; };
        for (ggml_type k : kvts) for (ggml_type v : kvts) { Case c = base(); c.kt = k; c.vt = v; emit(c); }
        for (ggml_type q : qts) for (ggml_type d : dsts) { Case c = base(); c.qt = q; c.dt = d; emit(c); }
        for (ggml_type mt : mts) for (int sinks : { 0, 1 }) for (ggml_type st : sts) {
            Case c = base(); c.mt = mt; c.sinks = sinks; c.st = st; emit(c);
        }
        for (ggml_type k : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) for (ggml_type q : { GGML_TYPE_F32, GGML_TYPE_F16 })
        for (int sinks : { 0, 1 }) for (int mask : { 0, 1 }) {
            Case c = base(); c.kt = c.vt = k; c.qt = q; c.sinks = sinks; c.mask = mask; emit(c);
        }
        { Case c = base(); c.rows = 1; emit(c); }
        { Case c = base(); c.rows = 1; c.sinks = 1; c.mask = 0; emit(c); }
    }

    // 5. Mask shapes and batch dimensions: broadcast, per-head and padded masks, sequence batches,
    //    broadcast K/V sequence dims, unequal K and V head counts.
    for (const auto & ms_ : { std::pair<int64_t, int64_t>{ 64, 1024 }, { 1, 2048 }, { 33, 300 } }) {
        for (int mask : { 0, 1, 2, 3, 4 }) for (int64_t n3 : { 1, 2 }) for (int64_t k3 : { 1, 2 }) for (int64_t v3 : { 1, 2 }) {
            if (n3 == 1 && (k3 != 1 || v3 != 1)) continue;
            Case c; c.m = ms_.first; c.s = ms_.second; c.mask = mask; c.n3 = n3; c.k3 = k3; c.v3 = v3;
            emit(c);
        }
        for (int64_t v2 : { 1, 2, 4, 8, 16 }) { Case c; c.m = ms_.first; c.s = ms_.second; c.v2 = v2; emit(c); }
        for (int sinks : { 0, 1 }) for (int mask : { 2, 3 }) {
            Case c; c.m = ms_.first; c.s = ms_.second; c.mask = mask; c.sinks = sinks; emit(c);
        }
    }

    // 6. Op params: softmax scale, ALiBi bias (the n_head -> n_head_log2, m0, m1 words), soft cap.
    for (float scale : { 0.125f, 0.0883883476f, 1.0f, 0.0f, 0.3f }) for (float bias : { 0.0f, 8.0f, 4.0f, 0.5f })
    for (float cap : { 0.0f, 30.0f, 50.0f, -30.0f, 0.25f })
    for (const auto & ms_ : { std::pair<int64_t, int64_t>{ 64, 1024 }, { 1, 100 } })
    for (int64_t nh : { 8, 16 }) {
        Case c; c.m = ms_.first; c.s = ms_.second; c.nh = nh; c.nkv = nh == 8 ? 4 : 8;
        c.scale = scale; c.max_bias = bias; c.softcap = cap;
        emit(c);
    }
    for (int64_t nh : { 1, 2, 3, 5, 6, 7, 9, 12, 20, 24, 31, 40, 48, 80, 100, 128, 255, 256, 512, 513 }) {
        for (const auto & ms_ : { std::pair<int64_t, int64_t>{ 64, 1024 }, { 1, 100 } }) {
            Case c; c.m = ms_.first; c.s = ms_.second; c.nh = nh; c.nkv = nh; c.max_bias = 8.0f; c.softcap = 30.0f;
            emit(c);
        }
    }

    // 7. Head dims and KV lengths at the edges: DK != DV, DK not a multiple of 8 or 64, very large
    //    DK (HMX at M = 1), KV lengths that overflow the 16-bit block count, one-row KV.
    const int64_t dks[] = { 8, 24, 40, 56, 64, 65, 72, 96, 120, 128, 160, 192, 256, 320, 512, 576 };
    const int64_t dvs[] = { 64, 128, 512 };
    for (int64_t dk : dks) for (int64_t dv : dvs) for (int64_t m : { 1, 32 }) for (int64_t s : { 1024, 8192 }) {
        Case c; c.dk = dk; c.dv = dv; c.m = m; c.s = s; emit(c);
    }
    for (int64_t s : { 1, 2, 65535, 65536, 65537, 100000, 4194303, 4194304, 4194305, 70000 * 64 }) for (int64_t m : { 1, 64 }) {
        Case c; c.m = m; c.s = s; emit(c);
    }
    for (int64_t m : { 1, 3, 31, 33, 100, 1000, 4096 }) for (int64_t s : { 5, 191, 8191 }) for (const Shape & sh : rep) {
        Case c; c.dk = sh.dk; c.dv = sh.dv; c.nh = sh.nh; c.nkv = sh.nkv; c.m = m; c.s = s; emit(c);
    }
    // GQA factors that do and do not divide a 32-row tile
    for (int64_t g : { 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64 }) for (int64_t m : { 1, 17 }) for (int64_t s : { 512, 4096 })
    for (uint32_t th : { 4u, 6u, 8u }) for (int64_t dk : { 64, 128 }) {
        Case c; c.dk = c.dv = dk; c.nkv = 4; c.nh = 4 * g; c.m = m; c.s = s; c.th = th; emit(c);
    }

    fprintf(stderr, "%d cases\n", g_cases);
    return 0;
}
