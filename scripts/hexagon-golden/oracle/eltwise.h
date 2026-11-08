// Helpers shared by the elementwise oracles (binary, unary, glu): layouts beyond `oracle_tensor`'s
// four, op names for the opcode check, and the print helpers. The unary and glu oracles run the
// node path the real host takes (`htp_opnode`, collapse, `htp_kparams_solve(sess, node)`) and check
// it against the direct entry points; binary has no collapse step and calls
// `htp_kparams_solve(sess, op, k)` directly.
#pragma once

#include "common.h"

#include <stdlib.h>

// Layouts of a dense 4-d tensor (`oracle_tensor` covers 0..3):
//   0 contiguous, 1 row-padded (16 spare bytes per row), 2 transposed view, 3 dim 2 inside dim 1,
//   4 rows three apart (a third of an in_proj-style strided view), 5 planes padded by 64 bytes.
static inline ggml_tensor mk(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, int kind = 0) {
    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    if (kind < 4) {
        return oracle_tensor(type, ne, kind);
    }
    const size_t ts  = ggml_type_size(type);
    const size_t row = ts * ne0;
    size_t       nb[4];
    nb[0] = ts;
    if (kind == 4) {
        nb[1] = row * 3; nb[2] = nb[1] * ne1; nb[3] = nb[2] * ne2;
    } else {
        nb[1] = row; nb[2] = nb[1] * ne1 + 64; nb[3] = nb[2] * ne2;
    }
    return oracle_tensor_nb(type, ne, nb);
}

#define ELTWISE_OPS(X)                                                                                  \
    X(HTP_OP_MUL) X(HTP_OP_ADD) X(HTP_OP_SUB) X(HTP_OP_DIV) X(HTP_OP_ADD_ID) X(HTP_OP_RMS_NORM)         \
    X(HTP_OP_RMS_NORM_MUL) X(HTP_OP_RMS_NORM_MUL_ADD) X(HTP_OP_UNARY_SILU) X(HTP_OP_UNARY_GELU)         \
    X(HTP_OP_UNARY_SIGMOID) X(HTP_OP_UNARY_EXP) X(HTP_OP_UNARY_NEG) X(HTP_OP_UNARY_SOFTPLUS)            \
    X(HTP_OP_UNARY_TANH) X(HTP_OP_UNARY_ABS) X(HTP_OP_UNARY_LOG) X(HTP_OP_UNARY_RELU)                   \
    X(HTP_OP_UNARY_STEP) X(HTP_OP_GLU_SWIGLU) X(HTP_OP_GLU_SWIGLU_OAI) X(HTP_OP_GLU_GEGLU)              \
    X(HTP_OP_GLU_GEGLU_QUICK) X(HTP_OP_SCALE) X(HTP_OP_SQR) X(HTP_OP_SQRT) X(HTP_OP_L2_NORM)            \
    X(HTP_OP_TRI) X(HTP_OP_NORM) X(HTP_OP_CLAMP) X(HTP_OP_LEAKY_RELU) X(HTP_OP_GLU_SWIGLU_CLAMP)        \
    X(HTP_OP_UNARY_GELU_ERF) X(HTP_OP_GLU_GEGLU_ERF) X(HTP_OP_UNARY_GELU_QUICK) X(HTP_OP_GLU_REGLU)     \
    X(HTP_OP_GLU_SIGMOID) X(HTP_OP_UNARY_CEIL) X(HTP_OP_UNARY_SGN) X(HTP_OP_CONV1D)                     \
    X(HTP_OP_UNARY_SNAKE) X(HTP_OP_UNARY_SIN) X(HTP_OP_UNARY_COS) X(HTP_OP_CONV_TRANSPOSE_1D)           \
    X(HTP_OP_UNARY_HARDSIGMOID) X(HTP_OP_UNARY_HARDSWISH) X(HTP_OP_UNARY_ELU)

// The opcode's C constant name: the Rust tests check it against `HtpOpCode::name`, which proves
// the Rust discriminants are the ones the DSP and host were built with.
static inline const char * eltwise_op_name(uint32_t op) {
#define X(c) case c: return #c;
    switch (op) {
        ELTWISE_OPS(X)
        default: return "?";
    }
#undef X
}

static inline void print_tensor_or_dash(const ggml_tensor * t) {
    if (t) {
        oracle_print_tensor(*t);
    } else {
        printf("-");
    }
}
