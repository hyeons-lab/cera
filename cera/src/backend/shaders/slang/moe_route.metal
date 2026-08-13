#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 73 "cera/src/backend/shaders/slang/moe_route.slang"
struct KernelContext_0
{
    packed_uint4 device* params_0;
    float device* logits_0;
    float device* bias_0;
    uint device* sel_expert_0;
    float device* sel_weight_0;
    array<float, int(256)> threadgroup* sh_prob_0;
    array<float, int(256)> threadgroup* sh_score_0;
};


#line 61
[[kernel]] void moe_route(uint3 lid_0 [[thread_position_in_threadgroup]], uint3 grp_0 [[threadgroup_position_in_grid]], packed_uint4 device* params_1 [[buffer(4)]], float device* logits_1 [[buffer(0)]], float device* bias_1 [[buffer(1)]], uint device* sel_expert_1 [[buffer(2)]], float device* sel_weight_1 [[buffer(3)]])
{

#line 61
    thread KernelContext_0 kernelContext_0;

#line 61
    (&kernelContext_0)->params_0 = params_1;

#line 61
    (&kernelContext_0)->logits_0 = logits_1;

#line 61
    (&kernelContext_0)->bias_0 = bias_1;

#line 61
    (&kernelContext_0)->sel_expert_0 = sel_expert_1;

#line 61
    (&kernelContext_0)->sel_weight_0 = sel_weight_1;

#line 61
    threadgroup array<float, int(256)> sh_prob_1;

#line 61
    (&kernelContext_0)->sh_prob_0 = &sh_prob_1;

#line 61
    threadgroup array<float, int(256)> sh_score_1;

#line 61
    (&kernelContext_0)->sh_score_0 = &sh_score_1;
    uint n_expert_0 = (uint4(*(params_1+int(0))) ).x;
    uint _S1 = min(min((uint4(*(params_1+int(0))) ).y, n_expert_0), 16U);

    uint tok_0 = grp_0.x;

    if(tok_0 >= ((uint4(*(params_1+int(0))) ).z))
    {

#line 68
        return;
    }

    uint _S2 = lid_0.x;

#line 71
    uint e_0 = _S2;

#line 71
    for(;;)
    {

#line 71
        if(e_0 < n_expert_0)
        {
        }
        else
        {

#line 71
            break;
        }

#line 72
        float p_0 = 1.0f / (1.0f + exp(- (&kernelContext_0)->logits_0[tok_0 * n_expert_0 + e_0]));
        (*(&kernelContext_0)->sh_prob_0)[e_0] = p_0;
        (*(&kernelContext_0)->sh_score_0)[e_0] = p_0 + (&kernelContext_0)->bias_0[e_0];

#line 71
        e_0 = e_0 + 32U;

#line 71
    }

#line 76
    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 85
    if(_S2 != 0U)
    {

#line 86
        return;
    }

#line 95
    thread array<uint, int(16)> chosen_0;
    thread array<float, int(16)> unnorm_w_0;

#line 96
    uint i_0 = 0U;
    for(;;)
    {

#line 97
        if(i_0 < 16U)
        {
        }
        else
        {

#line 97
            break;
        }

#line 98
        chosen_0[i_0] = 4294967295U;
        unnorm_w_0[i_0] = 0.0f;

#line 97
        i_0 = i_0 + 1U;

#line 97
    }

#line 97
    uint s_0 = 0U;

#line 97
    float sum_0 = 0.0f;

#line 103
    for(;;)
    {

#line 103
        if(s_0 < _S1)
        {
        }
        else
        {

#line 103
            break;
        }

#line 103
        bool have_0 = false;

#line 103
        float best_score_0 = 0.0f;

#line 103
        uint best_0 = 0U;

#line 103
        e_0 = 0U;

#line 108
        for(;;)
        {

#line 108
            if(e_0 < n_expert_0)
            {
            }
            else
            {

#line 108
                break;
            }

#line 108
            bool used_0 = false;

#line 108
            uint t_0 = 0U;

            for(;;)
            {

#line 110
                if(t_0 < s_0)
                {
                }
                else
                {

#line 110
                    break;
                }

#line 111
                if((chosen_0[t_0]) == e_0)
                {

#line 111
                    used_0 = true;

#line 111
                }

#line 110
                t_0 = t_0 + 1U;

#line 110
            }

#line 115
            if(used_0)
            {

#line 116
                e_0 = e_0 + 1U;

#line 108
                continue;
            }

#line 118
            float score_0 = (*(&kernelContext_0)->sh_score_0)[e_0];

#line 118
            bool _S3;
            if(!have_0)
            {

#line 119
                _S3 = true;

#line 119
            }
            else
            {

#line 119
                _S3 = score_0 > best_score_0;

#line 119
            }

#line 119
            uint best_1;

#line 119
            float best_score_1;

#line 119
            bool have_1;

#line 119
            if(_S3)
            {

#line 119
                have_1 = true;

#line 119
                best_score_1 = score_0;

#line 119
                best_1 = e_0;

#line 119
            }
            else
            {

#line 119
                have_1 = have_0;

#line 119
                best_score_1 = best_score_0;

#line 119
                best_1 = best_0;

#line 119
            }

#line 119
            have_0 = have_1;

#line 119
            best_score_0 = best_score_1;

#line 119
            best_0 = best_1;

#line 108
            e_0 = e_0 + 1U;

#line 108
        }

#line 126
        chosen_0[s_0] = best_0;
        float w_0 = (*(&kernelContext_0)->sh_prob_0)[best_0];
        unnorm_w_0[s_0] = (*(&kernelContext_0)->sh_prob_0)[best_0];
        *((&kernelContext_0)->sel_expert_0+(tok_0 * _S1 + s_0)) = best_0;
        float sum_1 = sum_0 + w_0;

#line 103
        s_0 = s_0 + 1U;

#line 103
        sum_0 = sum_1;

#line 103
    }

#line 135
    float _S4 = 1.0f / max(sum_0, 0.00006103515625f);

#line 135
    s_0 = 0U;
    for(;;)
    {

#line 136
        if(s_0 < _S1)
        {
        }
        else
        {

#line 136
            break;
        }

#line 137
        *((&kernelContext_0)->sel_weight_0+(tok_0 * _S1 + s_0)) = unnorm_w_0[s_0] * _S4;

#line 136
        s_0 = s_0 + 1U;

#line 136
    }


    return;
}

