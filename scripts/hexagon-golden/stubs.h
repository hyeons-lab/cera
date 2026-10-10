// Minimal stand-ins for the ggml types the llama.cpp Hexagon host code reads, so the real
// `ggml_hexagon_precompute_*` function text can be compiled on a desktop and its output used as
// golden data for the Rust ports. Only fields and helpers those functions touch are defined.
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>
#include <algorithm>
#include <set>

#define GGML_MAX_DIMS 4
#define GGML_ABORT(...) abort()
#define GGML_ASSERT(x) do { if (!(x)) abort(); } while (0)

enum ggml_type { GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1, GGML_TYPE_Q4_0 = 2, GGML_TYPE_Q8_0 = 8, GGML_TYPE_I32 = 26 };

struct ggml_tensor {
    enum ggml_type type;
    int64_t  ne[4];
    size_t   nb[4];
    struct ggml_tensor * src[10];
    int32_t  op_params[16];
    // the get_rows precompute asks whether the table was repacked
    const struct ggml_tensor * view_src;
    void *   buffer;
    void *   extra;
};

#define GGML_HEXAGON_TENSOR_REPACK 2
struct ggml_hexagon_tensor_extra { uint32_t flags; };
static inline bool ggml_backend_buffer_is_hexagon(void *) { return false; }

struct ggml_hexagon_session {
    int    n_threads;
    size_t vtcm_size;
    std::set<const ggml_tensor *> needs_repack;   // tables the host will repack before the op
};

static inline size_t  ggml_type_size(enum ggml_type t) { return t == GGML_TYPE_F16 ? 2 : 4; }  // F32/F16/I32 only
static inline int64_t ggml_blck_size(enum ggml_type)   { return 1; }
static inline int64_t ggml_nelements(const struct ggml_tensor * t) { return t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3]; }

// verbatim logic of ggml.c: ggml_is_contiguous_m_n(tensor, 0, GGML_MAX_DIMS)
static inline bool ggml_is_contiguous(const struct ggml_tensor * tensor) {
    const int m = 0, n = GGML_MAX_DIMS;
    size_t next_nb = ggml_type_size(tensor->type);
    if (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
        return false;
    }
    next_nb *= tensor->ne[0] / ggml_blck_size(tensor->type);
    for (int i = 1; i < n; i++) {
        if (i > m) {
            if (tensor->ne[i] != 1 && tensor->nb[i] != next_nb) {
                return false;
            }
            next_nb *= tensor->ne[i];
        } else {
            next_nb = tensor->ne[i] * tensor->nb[i];
        }
    }
    return true;
}
