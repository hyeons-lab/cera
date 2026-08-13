#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 60 "cera/src/backend/shaders/slang/moe_combine.slang"
struct KernelContext_0
{
    packed_uint4 device* params_0;
    float device* sel_weight_0;
    float device* z_0;
    float device* out_buf_0;
};


#line 42
[[kernel]] void moe_combine(uint3 lid_0 [[thread_position_in_threadgroup]], uint3 grp_0 [[threadgroup_position_in_grid]], packed_uint4 device* params_1 [[buffer(3)]], float device* sel_weight_1 [[buffer(1)]], float device* z_1 [[buffer(0)]], float device* out_buf_1 [[buffer(2)]])
{

#line 42
    thread KernelContext_0 kernelContext_0;

#line 42
    (&kernelContext_0)->params_0 = params_1;

#line 42
    (&kernelContext_0)->sel_weight_0 = sel_weight_1;

#line 42
    (&kernelContext_0)->z_0 = z_1;

#line 42
    (&kernelContext_0)->out_buf_0 = out_buf_1;
    uint hidden_0 = (uint4(*(params_1+int(0))) ).x;
    uint _S1 = (uint4(*(params_1+int(0))) ).y;
    uint n_tokens_0 = (uint4(*(params_1+int(0))) ).z;
    bool accumulate_0 = ((uint4(*(params_1+int(0))) ).w) != 0U;

    uint row_0 = grp_0.x * 256U + lid_0.x;
    uint tok_0 = grp_0.y;

#line 49
    bool _S2;
    if(row_0 >= hidden_0)
    {

#line 50
        _S2 = true;

#line 50
    }
    else
    {

#line 50
        _S2 = tok_0 >= n_tokens_0;

#line 50
    }

#line 50
    if(_S2)
    {

#line 51
        return;
    }

#line 51
    uint s_0 = 0U;

#line 51
    float acc_0 = 0.0f;



    for(;;)
    {

#line 55
        if(s_0 < _S1)
        {
        }
        else
        {

#line 55
            break;
        }

#line 56
        uint entry_0 = tok_0 * _S1 + s_0;
        float acc_1 = acc_0 + (&kernelContext_0)->sel_weight_0[entry_0] * (&kernelContext_0)->z_0[entry_0 * hidden_0 + row_0];

#line 55
        s_0 = s_0 + 1U;

#line 55
        acc_0 = acc_1;

#line 55
    }



    uint dst_0 = tok_0 * hidden_0 + row_0;
    float device* _S3 = (&kernelContext_0)->out_buf_0+dst_0;

#line 60
    if(accumulate_0)
    {

#line 60
        acc_0 = *((&kernelContext_0)->out_buf_0+dst_0) + acc_0;

#line 60
    }

#line 60
    *_S3 = acc_0;
    return;
}

