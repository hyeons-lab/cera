#include "common.h"

// The host's global options (ggml-hexagon.cpp), at their shipped defaults.
int    opt_arch       = 79;
size_t opt_ndev       = 1;
size_t opt_nhvx       = 0;
int    opt_nhmx       = 1;
size_t opt_vmem       = 0xc8000000ull;
int    opt_etm        = 0;
int    opt_verbose    = 0;
int    opt_profile    = 0;
bool   opt_hostbuf    = true;
bool   opt_dma64      = false;
int    opt_power_mode = HTP_POWER_MODE_BURST;

size_t opt_mbuf_dyn    = 512ul * 1024 * 1024;
size_t opt_mbuf_static = 1ul * 1024 * 1024 * 1024;
size_t opt_mbuf_total  = 0;

int opt_mm_select  = 2;
int opt_fa_select  = 2;
int opt_fa_split   = 1;
int opt_li_select  = 2;
int opt_gdn_select = 2;
int opt_ar_select  = 2;
int opt_ar_scatter = 1;

int opt_opbatch   = 2048;
int opt_opqueue   = 8;
int opt_optrace   = 0;
int opt_oppoll    = 0;
int opt_opfusion  = 1;
int opt_rt_repack = 1;

const char * htp_session_name(const struct ggml_hexagon_session *) { return "oracle"; }
size_t       htp_session_vtcm_size(const struct ggml_hexagon_session * s) { return s->vtcm_size; }
uint32_t     htp_session_n_threads(const struct ggml_hexagon_session * s) { return s->n_threads; }
uint32_t     htp_session_n_hmx(const struct ggml_hexagon_session * s) { return s->n_hmx; }
uint32_t     htp_session_mdev_count(const struct ggml_hexagon_session * s) { return s->mdev_count; }

ggml_tensor oracle_tensor_nb(ggml_type type, const int64_t * ne, const size_t * nb) {
    ggml_tensor t = {};
    t.type = type;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        t.ne[i] = ne[i];
        t.nb[i] = nb[i];
    }
    return t;
}

ggml_tensor oracle_tensor(ggml_type type, const int64_t * ne, int kind, bool tiled) {
    const size_t ts  = ggml_type_size(type);
    const size_t row = ggml_row_size(type, ne[0]);
    size_t nb[4];
    switch (kind) {
        case 1:  // row-padded
            nb[0] = ts; nb[1] = row + 16; nb[2] = nb[1] * ne[1]; nb[3] = nb[2] * ne[2];
            break;
        case 2:  // view of the transposed [ne1, ne0] matrix
            nb[0] = ts * ne[1]; nb[1] = ts; nb[2] = ts * ne[0] * ne[1]; nb[3] = nb[2] * ne[2];
            break;
        case 3:  // dim 2 stored inside dim 1
            nb[0] = ts; nb[2] = row; nb[1] = nb[2] * ne[2]; nb[3] = nb[1] * ne[1];
            break;
        default:
            nb[0] = ts; nb[1] = row; nb[2] = nb[1] * ne[1]; nb[3] = nb[2] * ne[2];
            break;
    }
    ggml_tensor t = oracle_tensor_nb(type, ne, nb);
    if (tiled) {
        t.extra = (void *) (uintptr_t) HTP_TENSOR_TILED;
    }
    return t;
}

void oracle_print_tensor(const ggml_tensor & t) {
    printf("%d %lld %lld %lld %lld %zu %zu %zu %zu", (int) t.type, (long long) t.ne[0], (long long) t.ne[1],
           (long long) t.ne[2], (long long) t.ne[3], t.nb[0], t.nb[1], t.nb[2], t.nb[3]);
}

void oracle_print_words(bool ok, const void * kparams, size_t bytes, int n) {
    if (!ok) {
        printf(" unsupported\n");
        return;
    }
    uint32_t w[32] = {};
    memcpy(w, kparams, bytes < sizeof(w) ? bytes : sizeof(w));
    for (int i = 0; i < n; i++) {
        printf(" %u", w[i]);
    }
    printf("\n");
}
