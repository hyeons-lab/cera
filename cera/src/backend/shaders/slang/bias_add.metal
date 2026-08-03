#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 32 "cera/src/backend/shaders/slang/bias_add.slang"
struct KernelContext_0
{
    packed_uint2 device* par_buf_0;
    float device* x_buf_0;
    float device* bias_buf_0;
};


#line 25
[[kernel]] void bias_add(uint3 gid_0 [[thread_position_in_grid]], packed_uint2 device* par_buf_1 [[buffer(2)]], float device* x_buf_1 [[buffer(0)]], float device* bias_buf_1 [[buffer(1)]])
{

#line 25
    thread KernelContext_0 kernelContext_0;

#line 25
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 25
    (&kernelContext_0)->x_buf_0 = x_buf_1;

#line 25
    (&kernelContext_0)->bias_buf_0 = bias_buf_1;
    uint i_0 = gid_0.x;

    uint dim_0 = (uint2(*(par_buf_1+int(0))) ).y;
    if(i_0 >= ((uint2(*(par_buf_1+int(0))) ).x))
    {

#line 30
        return;
    }
    float device* _S1 = (&kernelContext_0)->x_buf_0+i_0;

#line 32
    float _S2 = *((&kernelContext_0)->x_buf_0+i_0);

#line 32
    uint _S3 = i_0 % dim_0;

#line 32
    *_S1 = _S2 + (&kernelContext_0)->bias_buf_0[_S3];
    return;
}

