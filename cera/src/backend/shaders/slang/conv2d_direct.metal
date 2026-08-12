#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 100 "cera/src/backend/shaders/slang/conv2d_direct.slang"
struct KernelContext_0
{
    packed_uint4 device* par_buf_0;
    float device* bias_buf_0;
    float device* w_buf_0;
    float device* in_buf_0;
    float device* out_buf_0;
};


#line 44
[[kernel]] void conv2d_direct(uint3 gid_0 [[thread_position_in_grid]], packed_uint4 device* par_buf_1 [[buffer(4)]], float device* bias_buf_1 [[buffer(2)]], float device* w_buf_1 [[buffer(1)]], float device* in_buf_1 [[buffer(0)]], float device* out_buf_1 [[buffer(3)]])
{

#line 44
    thread KernelContext_0 kernelContext_0;

#line 44
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 44
    (&kernelContext_0)->bias_buf_0 = bias_buf_1;

#line 44
    (&kernelContext_0)->w_buf_0 = w_buf_1;

#line 44
    (&kernelContext_0)->in_buf_0 = in_buf_1;

#line 44
    (&kernelContext_0)->out_buf_0 = out_buf_1;
    uint in_ch_0 = (uint4(*(par_buf_1+int(0))) ).x;
    uint out_ch_0 = (uint4(*(par_buf_1+int(0))) ).y;
    uint _S1 = (uint4(*(par_buf_1+int(0))) ).z;
    uint _S2 = (uint4(*(par_buf_1+int(0))) ).w;
    uint _S3 = (uint4(*(par_buf_1+int(1))) ).x;
    uint _S4 = (uint4(*(par_buf_1+int(1))) ).y;
    uint str_h_0 = (uint4(*(par_buf_1+int(1))) ).z;
    uint str_w_0 = (uint4(*(par_buf_1+int(1))) ).w;
    uint pad_h_0 = (uint4(*(par_buf_1+int(2))) ).x;
    uint pad_w_0 = (uint4(*(par_buf_1+int(2))) ).y;

    uint w_out_0 = (uint4(*(par_buf_1+int(2))) ).w;
    uint groups_0 = (uint4(*(par_buf_1+int(3))) ).x;

    uint plane_out_0 = (uint4(*(par_buf_1+int(2))) ).z * w_out_0;

    uint idx_0 = gid_0.x;
    if(idx_0 >= (out_ch_0 * plane_out_0))
    {

#line 63
        return;
    }

    uint oc_0 = idx_0 / plane_out_0;
    uint rem_0 = idx_0 - oc_0 * plane_out_0;
    uint oh_0 = rem_0 / w_out_0;
    uint ow_0 = rem_0 - oh_0 * w_out_0;

    uint _S5 = in_ch_0 / groups_0;
    uint out_per_group_0 = out_ch_0 / groups_0;
    uint _S6 = oc_0 / out_per_group_0;



    int _S7 = int(oh_0 * str_h_0) - int(pad_h_0);
    int _S8 = int(ow_0 * str_w_0) - int(pad_w_0);

    float _S9 = (&kernelContext_0)->bias_buf_0[oc_0];

#line 80
    uint ic_local_0 = 0U;

#line 80
    float acc_0 = _S9;
    for(;;)
    {

#line 81
        if(ic_local_0 < _S5)
        {
        }
        else
        {

#line 81
            break;
        }
        uint _S10 = (oc_0 * _S5 + ic_local_0) * _S3 * _S4;
        uint _S11 = (_S6 * _S5 + ic_local_0) * _S1 * _S2;

#line 84
        uint ki_0 = 0U;

#line 84
        float acc_1 = acc_0;
        for(;;)
        {

#line 85
            if(ki_0 < _S3)
            {
            }
            else
            {

#line 85
                break;
            }

#line 86
            int ih_0 = _S7 + int(ki_0);

#line 86
            bool _S12;
            if(ih_0 < int(0))
            {

#line 87
                _S12 = true;

#line 87
            }
            else
            {

#line 87
                _S12 = ih_0 >= int(_S1);

#line 87
            }

#line 87
            float acc_2;

#line 87
            if(_S12)
            {

#line 87
                acc_2 = acc_1;
                ki_0 = ki_0 + 1U;

#line 88
                acc_1 = acc_2;

#line 85
                continue;
            }



            uint _S13 = _S11 + uint(ih_0) * _S2;

#line 90
            uint kj_0 = 0U;

#line 90
            acc_2 = acc_1;
            for(;;)
            {

#line 91
                if(kj_0 < _S4)
                {
                }
                else
                {

#line 91
                    break;
                }

#line 92
                int iw_0 = _S8 + int(kj_0);

#line 92
                bool _S14;
                if(iw_0 < int(0))
                {

#line 93
                    _S14 = true;

#line 93
                }
                else
                {

#line 93
                    _S14 = iw_0 >= int(_S2);

#line 93
                }

#line 93
                if(_S14)
                {

#line 94
                    kj_0 = kj_0 + 1U;

#line 91
                    continue;
                }

#line 91
                acc_2 = acc_2 + (&kernelContext_0)->w_buf_0[_S10 + ki_0 * _S4 + kj_0] * (&kernelContext_0)->in_buf_0[_S13 + uint(iw_0)];

#line 91
                kj_0 = kj_0 + 1U;

#line 91
            }

#line 85
            ki_0 = ki_0 + 1U;

#line 85
            acc_1 = acc_2;

#line 85
        }

#line 81
        ic_local_0 = ic_local_0 + 1U;

#line 81
        acc_0 = acc_1;

#line 81
    }

#line 100
    *((&kernelContext_0)->out_buf_0+idx_0) = acc_0;
    return;
}

