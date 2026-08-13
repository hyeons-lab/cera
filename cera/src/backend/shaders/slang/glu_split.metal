#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 43 "cera/src/backend/shaders/slang/glu_split.slang"
struct KernelContext_0
{
    packed_uint4 device* par_buf_0;
    float device* src_buf_0;
    float device* dst_buf_0;
};


#line 27
[[kernel]] void glu_split(uint3 gid_0 [[thread_position_in_grid]], packed_uint4 device* par_buf_1 [[buffer(2)]], float device* src_buf_1 [[buffer(0)]], float device* dst_buf_1 [[buffer(1)]])
{

#line 27
    thread KernelContext_0 kernelContext_0;

#line 27
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 27
    (&kernelContext_0)->src_buf_0 = src_buf_1;

#line 27
    (&kernelContext_0)->dst_buf_0 = dst_buf_1;

    uint n_0 = (uint4(*(par_buf_1+int(0))) ).y;


    uint idx_0 = gid_0.x;
    if(idx_0 >= ((uint4(*(par_buf_1+int(0))) ).x * n_0))
    {

#line 34
        return;
    }

    uint r_0 = idx_0 / n_0;
    uint c_0 = idx_0 - r_0 * n_0;
    uint base_0 = r_0 * 2U * n_0;



    *((&kernelContext_0)->dst_buf_0+idx_0) = (&kernelContext_0)->src_buf_0[base_0 + c_0] / (1.0f + exp(- (&kernelContext_0)->src_buf_0[base_0 + n_0 + c_0]));
    return;
}

