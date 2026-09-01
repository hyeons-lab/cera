#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 74 "cera/src/backend/shaders/slang/moe_route.slang"
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
        float p_0 = 1.0f / (1.0f + exp(- clamp((&kernelContext_0)->logits_0[tok_0 * n_expert_0 + e_0], -80.0f, 80.0f)));
        (*(&kernelContext_0)->sh_prob_0)[e_0] = p_0;
        (*(&kernelContext_0)->sh_score_0)[e_0] = p_0 + (&kernelContext_0)->bias_0[e_0];

#line 71
        e_0 = e_0 + 32U;

#line 71
    }

#line 77
    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 86
    if(_S2 != 0U)
    {

#line 87
        return;
    }

#line 96
    thread array<uint, int(16)> chosen_0;
    thread array<float, int(16)> unnorm_w_0;

#line 97
    uint i_0 = 0U;
    for(;;)
    {

#line 98
        if(i_0 < 16U)
        {
        }
        else
        {

#line 98
            break;
        }

#line 99
        chosen_0[i_0] = 4294967295U;
        unnorm_w_0[i_0] = 0.0f;

#line 98
        i_0 = i_0 + 1U;

#line 98
    }

#line 98
    uint s_0 = 0U;

#line 98
    float sum_0 = 0.0f;

#line 104
    for(;;)
    {

#line 104
        if(s_0 < _S1)
        {
        }
        else
        {

#line 104
            break;
        }

#line 104
        bool have_0 = false;

#line 104
        float best_score_0 = 0.0f;

#line 104
        uint best_0 = 0U;

#line 104
        e_0 = 0U;

#line 109
        for(;;)
        {

#line 109
            if(e_0 < n_expert_0)
            {
            }
            else
            {

#line 109
                break;
            }

#line 109
            bool used_0 = false;

#line 109
            uint t_0 = 0U;

            for(;;)
            {

#line 111
                if(t_0 < s_0)
                {
                }
                else
                {

#line 111
                    break;
                }

#line 112
                if((chosen_0[t_0]) == e_0)
                {

#line 112
                    used_0 = true;

#line 112
                }

#line 111
                t_0 = t_0 + 1U;

#line 111
            }

#line 116
            if(used_0)
            {

#line 117
                e_0 = e_0 + 1U;

#line 109
                continue;
            }

#line 119
            float score_0 = (*(&kernelContext_0)->sh_score_0)[e_0];

#line 119
            bool _S3;
            if(!have_0)
            {

#line 120
                _S3 = true;

#line 120
            }
            else
            {

#line 120
                _S3 = score_0 > best_score_0;

#line 120
            }

#line 120
            uint best_1;

#line 120
            float best_score_1;

#line 120
            bool have_1;

#line 120
            if(_S3)
            {

#line 120
                have_1 = true;

#line 120
                best_score_1 = score_0;

#line 120
                best_1 = e_0;

#line 120
            }
            else
            {

#line 120
                have_1 = have_0;

#line 120
                best_score_1 = best_score_0;

#line 120
                best_1 = best_0;

#line 120
            }

#line 120
            have_0 = have_1;

#line 120
            best_score_0 = best_score_1;

#line 120
            best_0 = best_1;

#line 109
            e_0 = e_0 + 1U;

#line 109
        }

#line 127
        chosen_0[s_0] = best_0;
        float w_0 = (*(&kernelContext_0)->sh_prob_0)[best_0];
        unnorm_w_0[s_0] = (*(&kernelContext_0)->sh_prob_0)[best_0];
        *((&kernelContext_0)->sel_expert_0+(tok_0 * _S1 + s_0)) = best_0;
        float sum_1 = sum_0 + w_0;

#line 104
        s_0 = s_0 + 1U;

#line 104
        sum_0 = sum_1;

#line 104
    }

#line 136
    float _S4 = 1.0f / max(sum_0, 0.00006103515625f);

#line 136
    s_0 = 0U;
    for(;;)
    {

#line 137
        if(s_0 < _S1)
        {
        }
        else
        {

#line 137
            break;
        }

#line 138
        *((&kernelContext_0)->sel_weight_0+(tok_0 * _S1 + s_0)) = unnorm_w_0[s_0] * _S4;

#line 137
        s_0 = s_0 + 1U;

#line 137
    }


    return;
}

