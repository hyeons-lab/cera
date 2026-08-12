#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 45 "cera/src/backend/shaders/slang/transpose_blocked.slang"
struct KernelContext_0
{
    packed_uint4 device* par_buf_0;
    float device* dst_buf_0;
    float device* src_buf_0;
};


#line 28
[[kernel]] void transpose_blocked(uint3 gid_0 [[thread_position_in_grid]], packed_uint4 device* par_buf_1 [[buffer(2)]], float device* dst_buf_1 [[buffer(1)]], float device* src_buf_1 [[buffer(0)]])
{

#line 28
    thread KernelContext_0 kernelContext_0;

#line 28
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 28
    (&kernelContext_0)->dst_buf_0 = dst_buf_1;

#line 28
    (&kernelContext_0)->src_buf_0 = src_buf_1;
    uint a_dim_0 = (uint4(*(par_buf_1+int(0))) ).x;

    uint k_dim_0 = (uint4(*(par_buf_1+int(0))) ).z;

    uint bk_0 = (uint4(*(par_buf_1+int(0))) ).y * k_dim_0;

    uint idx_0 = gid_0.x;
    if(idx_0 >= (a_dim_0 * bk_0))
    {

#line 37
        return;
    }

    uint a_0 = idx_0 / bk_0;
    uint rem_0 = idx_0 - a_0 * bk_0;
    uint b_0 = rem_0 / k_dim_0;


    *((&kernelContext_0)->dst_buf_0+((b_0 * a_dim_0 + a_0) * k_dim_0 + (rem_0 - b_0 * k_dim_0))) = (&kernelContext_0)->src_buf_0[idx_0];
    return;
}

