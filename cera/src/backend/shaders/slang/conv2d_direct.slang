// Direct (no im2col) 2D convolution, one thread per output element.
//
// Covers every convolution the LFM2A audio encoder needs, on both GPU backends:
//
//   - the subsampling stem's regular 3x3 s2 p1 (1 -> 256), its two depthwise
//     3x3 s2 p1 layers (`groups == in_ch`), and its two pointwise 1x1 layers,
//   - each Conformer block's depthwise conv1d over the full time axis, which is
//     this kernel with `kh = 1`, `h_in = 1`, `pad_h = 0`, `stride_h = 1`.
//
// The 1D case falls out of the 2D one because `cpu::conv1d` and `cpu::conv2d`
// agree on weight layout: both index `[oc][ic_local][k...]` with the spatial
// axes fastest, so a `[channels, k]` conv1d weight is bit-identical to a
// `[out_ch, 1, 1, k]` conv2d weight. One kernel, one parity case, no separate
// `conv1d_conformer`.
//
// Padding is expressed as a **low-side** amount plus host-computed output dims,
// rather than a symmetric pad the kernel expands itself. That keeps the
// asymmetric split `cpu::conformer_conv_module_forward` uses for even kernel
// sizes (`pad_left = (k-1)/2`, remainder on the right) expressible, instead of
// silently shifting the convolution by half a tap.
//
//   binding/buffer 0: input   f32, read       [in_ch][h_in][w_in]
//   binding/buffer 1: weight  f32, read       [out_ch][in_per_group][kh][kw]
//   binding/buffer 2: bias    f32, read       [out_ch] (always bound; pass zeros
//                                             for a bias-free conv)
//   binding/buffer 3: out     f32, write      [out_ch][h_out][w_out]
//   binding/buffer 4: params  uint4 x 4:
//       [0] = (in_ch, out_ch, h_in, w_in)
//       [1] = (kh, kw, stride_h, stride_w)
//       [2] = (pad_h_lo, pad_w_lo, h_out, w_out)
//       [3] = (groups, _, _, _)
//
// Dispatch: one thread per output element, ceil(out_ch*h_out*w_out / 256)
// workgroups of 256.

[[vk::binding(0)]] StructuredBuffer<float>   in_buf   : register(t0);
[[vk::binding(1)]] StructuredBuffer<float>   w_buf    : register(t1);
[[vk::binding(2)]] StructuredBuffer<float>   bias_buf : register(t2);
[[vk::binding(3)]] RWStructuredBuffer<float> out_buf  : register(u3);
[[vk::binding(4)]] StructuredBuffer<uint4>   par_buf  : register(t4);

[shader("compute")]
[numthreads(256, 1, 1)]
void conv2d_direct(uint3 gid : SV_DispatchThreadID) {
    uint in_ch  = par_buf[0].x;
    uint out_ch = par_buf[0].y;
    uint h_in   = par_buf[0].z;
    uint w_in   = par_buf[0].w;
    uint kh     = par_buf[1].x;
    uint kw     = par_buf[1].y;
    uint str_h  = par_buf[1].z;
    uint str_w  = par_buf[1].w;
    uint pad_h  = par_buf[2].x;
    uint pad_w  = par_buf[2].y;
    uint h_out  = par_buf[2].z;
    uint w_out  = par_buf[2].w;
    uint groups = par_buf[3].x;

    uint plane_out = h_out * w_out;
    uint total = out_ch * plane_out;
    uint idx = gid.x;
    if (idx >= total) {
        return;
    }

    uint oc = idx / plane_out;
    uint rem = idx - oc * plane_out;
    uint oh = rem / w_out;
    uint ow = rem - oh * w_out;

    uint in_per_group = in_ch / groups;
    uint out_per_group = out_ch / groups;
    uint g = oc / out_per_group;

    // Signed so the low-side pad can push the tap before the input's first
    // element; an unsigned subtraction would wrap and read out of bounds.
    int ih_base = int(oh * str_h) - int(pad_h);
    int iw_base = int(ow * str_w) - int(pad_w);

    float acc = bias_buf[oc];
    for (uint ic_local = 0u; ic_local < in_per_group; ic_local++) {
        uint ic = g * in_per_group + ic_local;
        uint w_row = (oc * in_per_group + ic_local) * kh * kw;
        uint in_plane = ic * h_in * w_in;
        for (uint ki = 0u; ki < kh; ki++) {
            int ih = ih_base + int(ki);
            if (ih < 0 || ih >= int(h_in)) {
                continue;
            }
            uint in_row = in_plane + uint(ih) * w_in;
            for (uint kj = 0u; kj < kw; kj++) {
                int iw = iw_base + int(kj);
                if (iw < 0 || iw >= int(w_in)) {
                    continue;
                }
                acc += w_buf[w_row + ki * kw + kj] * in_buf[in_row + uint(iw)];
            }
        }
    }
    out_buf[idx] = acc;
}
