#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 40 "cera/src/backend/shaders/slang/activations.slang"
struct KernelContext_0
{
    packed_uint2 device* par_buf_0;
    float device* x_buf_0;
};


#line 35
[[kernel]] void relu_inplace(uint3 gid_0 [[thread_position_in_grid]], packed_uint2 device* par_buf_1 [[buffer(1)]], float device* x_buf_1 [[buffer(0)]])
{

#line 35
    thread KernelContext_0 kernelContext_0;

#line 35
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 35
    (&kernelContext_0)->x_buf_0 = x_buf_1;
    uint i_0 = gid_0.x;
    if(i_0 >= ((uint2(*(par_buf_1+int(0))) ).x))
    {

#line 38
        return;
    }
    *((&kernelContext_0)->x_buf_0+i_0) = max(*((&kernelContext_0)->x_buf_0+i_0), 0.0f);
    return;
}


[[kernel]] void silu_inplace(uint3 gid_1 [[thread_position_in_grid]], packed_uint2 device* par_buf_2 [[buffer(1)]], float device* x_buf_2 [[buffer(0)]])
{

#line 45
    thread KernelContext_0 kernelContext_1;

#line 45
    (&kernelContext_1)->par_buf_0 = par_buf_2;

#line 45
    (&kernelContext_1)->x_buf_0 = x_buf_2;
    uint i_1 = gid_1.x;
    if(i_1 >= ((uint2(*(par_buf_2+int(0))) ).x))
    {

#line 48
        return;
    }
    float v_0 = clamp(*((&kernelContext_1)->x_buf_0+i_1), -80.0f, 80.0f);
    *((&kernelContext_1)->x_buf_0+i_1) = v_0 / (1.0f + exp(- v_0));
    return;
}


[[kernel]] void gelu_erf_inplace(uint3 gid_2 [[thread_position_in_grid]], packed_uint2 device* par_buf_3 [[buffer(1)]], float device* x_buf_3 [[buffer(0)]])
{

#line 56
    thread KernelContext_0 kernelContext_2;

#line 56
    (&kernelContext_2)->par_buf_0 = par_buf_3;

#line 56
    (&kernelContext_2)->x_buf_0 = x_buf_3;
    uint i_2 = gid_2.x;
    if(i_2 >= ((uint2(*(par_buf_3+int(0))) ).x))
    {

#line 59
        return;
    }
    float device* _S1 = (&kernelContext_2)->x_buf_0+i_2;

#line 61
    float v_1 = *_S1;


    float a_0 = abs(*_S1 * 0.70710676908493042f);

#line 64
    float sign_0;
    if((*_S1) < 0.0f)
    {

#line 65
        sign_0 = -1.0f;

#line 65
    }
    else
    {

#line 65
        sign_0 = 1.0f;

#line 65
    }
    float t_0 = 1.0f / (1.0f + 0.32759109139442444f * a_0);

#line 71
    *((&kernelContext_2)->x_buf_0+i_2) = 0.5f * v_1 * (1.0f + sign_0 * (1.0f - ((((1.06140542030334473f * t_0 - 1.45315194129943848f) * t_0 + 1.42141366004943848f) * t_0 - 0.28449669480323792f) * t_0 + 0.25482958555221558f) * t_0 * exp(- a_0 * a_0)));
    return;
}

