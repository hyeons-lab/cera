#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 36 "cera/src/backend/shaders/slang/exp_polar.slang"
struct KernelContext_0
{
    packed_uint2 device* par_buf_0;
    float device* spec_buf_0;
    float device* out_buf_0;
};


#line 23
[[kernel]] void exp_polar(uint3 gid_0 [[thread_position_in_grid]], packed_uint2 device* par_buf_1 [[buffer(2)]], float device* spec_buf_1 [[buffer(0)]], float device* out_buf_1 [[buffer(1)]])
{

#line 23
    thread KernelContext_0 kernelContext_0;

#line 23
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 23
    (&kernelContext_0)->spec_buf_0 = spec_buf_1;

#line 23
    (&kernelContext_0)->out_buf_0 = out_buf_1;
    uint bins_0 = (uint2(*(par_buf_1+int(0))) ).y;

    uint i_0 = gid_0.x;
    if(i_0 >= ((uint2(*(par_buf_1+int(0))) ).x * bins_0))
    {

#line 28
        return;
    }
    uint frame_0 = i_0 / bins_0;
    uint j_0 = i_0 % bins_0;
    uint base_0 = frame_0 * 2U * bins_0;
    uint _S1 = base_0 + j_0;
    uint _S2 = base_0 + bins_0 + j_0;

#line 34
    float angle_0 = (&kernelContext_0)->spec_buf_0[_S2];
    float mag_0 = exp((&kernelContext_0)->spec_buf_0[_S1]);
    *((&kernelContext_0)->out_buf_0+_S1) = mag_0 * cos(angle_0);
    *((&kernelContext_0)->out_buf_0+_S2) = mag_0 * sin(angle_0);
    return;
}

