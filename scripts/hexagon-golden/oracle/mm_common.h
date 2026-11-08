// Helpers shared by the matmul oracles (`matmul*.cpp`, `mmnx.cpp`, `mmidnx.cpp`): session
// configurations, tensor printing with the tiled flag, and the params words of
// `htp_mm_kernel_params`.
//
// cera uploads its own tiled weights and never relies on the host's runtime repack, so these
// oracles run with `opt_rt_repack = 0`: a weight is tiled exactly when its `extra` flag says so
// (`oracle_tensor(..., tiled = true)`).
#pragma once

#include "common.h"

extern int opt_rt_repack;

// One session: DSP threads, VTCM bytes, HMX units, devices.
struct MmCfg {
    uint32_t th;
    size_t   vt;
    uint32_t hmx;
    uint32_t mdev;
};

static const MmCfg MM_BASE = { 8, 8u << 20, 1, 1 };

static inline ggml_hexagon_session mm_sess(const MmCfg & c) {
    ggml_hexagon_session s;
    s.n_threads  = c.th;
    s.vtcm_size  = c.vt;
    s.n_hmx      = c.hmx;
    s.mdev_count = c.mdev;
    return s;
}

static inline void mm_print_cfg(const MmCfg & c) {
    printf("%u %zu %u %u", c.th, c.vt, c.hmx, c.mdev);
}

// "type ne0 ne1 ne2 ne3 nb0 nb1 nb2 nb3 flags"
static inline void mm_print_t(const ggml_tensor & t) {
    oracle_print_tensor(t);
    printf(" %u", (uint32_t) (uintptr_t) t.extra);
}

static inline bool mm_same_t(const ggml_tensor & a, const ggml_tensor & b) {
    return a.type == b.type && a.extra == b.extra && memcmp(a.ne, b.ne, sizeof(a.ne)) == 0 &&
           memcmp(a.nb, b.nb, sizeof(a.nb)) == 0;
}

// The tensor, or "=" when the node kept the original (so a collapse that did nothing costs
// one character).
static inline void mm_print_t_or_same(const ggml_tensor & t, const ggml_tensor & orig) {
    if (mm_same_t(t, orig)) {
        printf("=");
    } else {
        mm_print_t(t);
    }
}

// The 23 words of `htp_mm_kernel_params` (the rest of the blob is zero), or "unsupported".
static inline void mm_print_words(bool ok, const htp_mm_kernel_params & kp) {
    if (!ok) {
        printf(" unsupported");
        return;
    }
    uint32_t w[32] = {};
    memcpy(w, &kp, sizeof(kp));
    for (size_t i = 0; i < sizeof(kp) / 4; i++) {
        printf(" %u", w[i]);
    }
}

static inline ggml_tensor mm_w(ggml_type t, int64_t k, int64_t n, int64_t ne2 = 1, bool tiled = true) {
    const int64_t ne[4] = { k, n, ne2, 1 };
    return oracle_tensor(t, ne, 0, tiled);
}

// F32 activation / output of `rows` rows, `kind` as `oracle_tensor`.
static inline ggml_tensor mm_f(int64_t w, int64_t rows, int64_t ne2 = 1, int kind = 0, ggml_type t = GGML_TYPE_F32) {
    const int64_t ne[4] = { w, rows, ne2, 1 };
    return oracle_tensor(t, ne, kind);
}

static const size_t MM_VTCMS[] = { 8u << 20, 4u << 20, 1u << 20, 512u << 10, 128u << 10, 64u << 10, 0 };
