#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 33 "cera/src/backend/shaders/slang/elementwise.slang"
struct KernelContext_0
{
    packed_uint2 device* par_buf_0;
    float device* a_buf_0;
    float device* b_buf_0;
};


#line 28
[[kernel]] void add_inplace(uint3 gid_0 [[thread_position_in_grid]], packed_uint2 device* par_buf_1 [[buffer(2)]], float device* a_buf_1 [[buffer(0)]], float device* b_buf_1 [[buffer(1)]])
{

#line 28
    thread KernelContext_0 kernelContext_0;

#line 28
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 28
    (&kernelContext_0)->a_buf_0 = a_buf_1;

#line 28
    (&kernelContext_0)->b_buf_0 = b_buf_1;
    uint i_0 = gid_0.x;
    if(i_0 >= ((uint2(*(par_buf_1+int(0))) ).x))
    {

#line 31
        return;
    }
    *((&kernelContext_0)->a_buf_0+i_0) = *((&kernelContext_0)->a_buf_0+i_0) + (&kernelContext_0)->b_buf_0[i_0];
    return;
}


#line 41
[[kernel]] void scaled_add_inplace(uint3 gid_1 [[thread_position_in_grid]], packed_uint2 device* par_buf_2 [[buffer(2)]], float device* a_buf_2 [[buffer(0)]], float device* b_buf_2 [[buffer(1)]])
{

#line 41
    thread KernelContext_0 kernelContext_1;

#line 41
    (&kernelContext_1)->par_buf_0 = par_buf_2;

#line 41
    (&kernelContext_1)->a_buf_0 = a_buf_2;

#line 41
    (&kernelContext_1)->b_buf_0 = b_buf_2;
    uint i_1 = gid_1.x;
    if(i_1 >= ((uint2(*(par_buf_2+int(0))) ).x))
    {

#line 44
        return;
    }
    *((&kernelContext_1)->a_buf_0+i_1) = *((&kernelContext_1)->a_buf_0+i_1) + (as_type<float>(((uint2(*((&kernelContext_1)->par_buf_0+int(0))) ).y))) * (&kernelContext_1)->b_buf_0[i_1];
    return;
}


[[kernel]] void mul_inplace(uint3 gid_2 [[thread_position_in_grid]], packed_uint2 device* par_buf_3 [[buffer(2)]], float device* a_buf_3 [[buffer(0)]], float device* b_buf_3 [[buffer(1)]])
{

#line 51
    thread KernelContext_0 kernelContext_2;

#line 51
    (&kernelContext_2)->par_buf_0 = par_buf_3;

#line 51
    (&kernelContext_2)->a_buf_0 = a_buf_3;

#line 51
    (&kernelContext_2)->b_buf_0 = b_buf_3;
    uint i_2 = gid_2.x;
    if(i_2 >= ((uint2(*(par_buf_3+int(0))) ).x))
    {

#line 54
        return;
    }
    *((&kernelContext_2)->a_buf_0+i_2) = *((&kernelContext_2)->a_buf_0+i_2) * (&kernelContext_2)->b_buf_0[i_2];
    return;
}


[[kernel]] void silu_mul_inplace(uint3 gid_3 [[thread_position_in_grid]], packed_uint2 device* par_buf_4 [[buffer(2)]], float device* a_buf_4 [[buffer(0)]], float device* b_buf_4 [[buffer(1)]])
{

#line 61
    thread KernelContext_0 kernelContext_3;

#line 61
    (&kernelContext_3)->par_buf_0 = par_buf_4;

#line 61
    (&kernelContext_3)->a_buf_0 = a_buf_4;

#line 61
    (&kernelContext_3)->b_buf_0 = b_buf_4;
    uint i_3 = gid_3.x;
    if(i_3 >= ((uint2(*(par_buf_4+int(0))) ).x))
    {

#line 64
        return;
    }
    float g_0 = clamp(*((&kernelContext_3)->a_buf_0+i_3), -80.0f, 80.0f);
    *((&kernelContext_3)->a_buf_0+i_3) = g_0 / (1.0f + exp(- g_0)) * (&kernelContext_3)->b_buf_0[i_3];
    return;
}

