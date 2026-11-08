// Shared pieces of the golden-parameter oracle: a program that links llama.cpp's real Hexagon host
// solvers (`htp-kparams.cpp`, `htp-opnode.cpp`, `htp-repack.cpp`, built unmodified from a llama.cpp
// checkout) against ggml-base on the desktop and prints what they compute for a table of cases.
// The Rust ports are then pinned to those words, so a port is checked against the code it mirrors.
#pragma once

#include "htp-kparams.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

// What the solvers read from a session (`ggml_hexagon_session` is opaque to them: see the
// `htp_session_*` accessors in common.cpp). Defaults mirror a real S25 Ultra session.
struct ggml_hexagon_session {
    uint32_t n_threads  = 6;
    size_t   vtcm_size  = 8u << 20;
    uint32_t n_hmx      = 1;
    uint32_t mdev_count = 1;
};

// A dense, 4-d ggml tensor of `type` with extents `ne` and the strides `kind` selects:
//   0 contiguous, 1 row-padded (16 spare bytes per row), 2 transposed view (dim 0 strided),
//   3 dim 2 stored inside dim 1 (a permuted view).
// `tiled` marks a weight as already in the DSP's tiled layout (the host's `extra` flag).
// Strides are in bytes; blocks are counted by ggml's own `ggml_row_size`.
ggml_tensor oracle_tensor(ggml_type type, const int64_t * ne, int kind = 0, bool tiled = false);

// Same tensor with an explicit byte stride per dimension.
ggml_tensor oracle_tensor_nb(ggml_type type, const int64_t * ne, const size_t * nb);

// "type ne0 ne1 ne2 ne3 nb0 nb1 nb2 nb3", the layout the Rust tests parse back.
void oracle_print_tensor(const ggml_tensor & t);

// First `n` words of a kernel-params blob, then a newline; an unsupported case prints
// "unsupported" instead. Pass n = 32 for the whole `[i32; 32]` blob.
void oracle_print_words(bool ok, const void * kparams, size_t bytes, int n = 32);

