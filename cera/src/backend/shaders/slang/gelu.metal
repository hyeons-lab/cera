#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 34 "cera/src/backend/shaders/slang/gelu.slang"
struct KernelContext_0
{
    packed_uint2 device* par_buf_0;
    float device* x_buf_0;
};


#line 28
[[kernel]] void gelu_inplace(uint3 gid_0 [[thread_position_in_grid]], packed_uint2 device* par_buf_1 [[buffer(1)]], float device* x_buf_1 [[buffer(0)]])
{

#line 28
    thread KernelContext_0 kernelContext_0;

#line 28
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 28
    (&kernelContext_0)->x_buf_0 = x_buf_1;
    uint i_0 = gid_0.x;

    if(i_0 >= ((uint2(*(par_buf_1+int(0))) ).x))
    {

#line 32
        return;
    }
    float device* _S1 = (&kernelContext_0)->x_buf_0+i_0;

#line 40
    *((&kernelContext_0)->x_buf_0+i_0) = 0.5f * *_S1 * (1.0f + tanh(clamp(0.79788458347320557f * (*_S1 + 0.04471499845385551f * *_S1 * *_S1 * *_S1), -15.0f, 15.0f)));
    return;
}

