#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 42 "cera/src/backend/shaders/slang/chan_affine_silu.slang"
struct KernelContext_0
{
    packed_uint4 device* par_buf_0;
    float device* x_buf_0;
    float device* w_buf_0;
    float device* b_buf_0;
};


#line 31
[[kernel]] void chan_affine_silu(uint3 gid_0 [[thread_position_in_grid]], packed_uint4 device* par_buf_1 [[buffer(3)]], float device* x_buf_1 [[buffer(0)]], float device* w_buf_1 [[buffer(1)]], float device* b_buf_1 [[buffer(2)]])
{

#line 31
    thread KernelContext_0 kernelContext_0;

#line 31
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 31
    (&kernelContext_0)->x_buf_0 = x_buf_1;

#line 31
    (&kernelContext_0)->w_buf_0 = w_buf_1;

#line 31
    (&kernelContext_0)->b_buf_0 = b_buf_1;

    uint t_0 = (uint4(*(par_buf_1+int(0))) ).y;


    uint idx_0 = gid_0.x;
    if(idx_0 >= ((uint4(*(par_buf_1+int(0))) ).x * t_0))
    {

#line 38
        return;
    }

    uint c_0 = idx_0 / t_0;
    float v_0 = clamp(*((&kernelContext_0)->x_buf_0+idx_0) * (&kernelContext_0)->w_buf_0[c_0] + (&kernelContext_0)->b_buf_0[c_0], -80.0f, 80.0f);
    *((&kernelContext_0)->x_buf_0+idx_0) = v_0 / (1.0f + exp(- v_0));
    return;
}

