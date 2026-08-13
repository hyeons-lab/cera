#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 71 "cera/src/backend/shaders/slang/moe_route.slang"
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


#line 60
[[kernel]] void moe_route(uint3 lid_0 [[thread_position_in_threadgroup]], uint3 grp_0 [[threadgroup_position_in_grid]], packed_uint4 device* params_1 [[buffer(4)]], float device* logits_1 [[buffer(0)]], float device* bias_1 [[buffer(1)]], uint device* sel_expert_1 [[buffer(2)]], float device* sel_weight_1 [[buffer(3)]])
{

#line 60
    thread KernelContext_0 kernelContext_0;

#line 60
    (&kernelContext_0)->params_0 = params_1;

#line 60
    (&kernelContext_0)->logits_0 = logits_1;

#line 60
    (&kernelContext_0)->bias_0 = bias_1;

#line 60
    (&kernelContext_0)->sel_expert_0 = sel_expert_1;

#line 60
    (&kernelContext_0)->sel_weight_0 = sel_weight_1;

#line 60
    threadgroup array<float, int(256)> sh_prob_1;
    threadgroup array<float, int(256)> sh_score_1;

#line 60
    (&kernelContext_0)->sh_prob_0 = &sh_prob_1;
    (&kernelContext_0)->sh_score_0 = &sh_score_1;
    uint n_expert_0 = (uint4(*(params_1+int(0))) ).x;
    uint _S1 = min(min((uint4(*(params_1+int(0))) ).y, n_expert_0), 16U);

    uint tok_0 = grp_0.x;

    if(tok_0 >= ((uint4(*(params_1+int(0))) ).z))
    {

#line 67
        return;
    }

    uint _S2 = lid_0.x;

#line 70
    uint e_0 = _S2;

#line 70
    for(;;)
    {

#line 70
        if(e_0 < n_expert_0)
        {
        }
        else
        {

#line 70
            break;
        }

#line 71
        float p_0 = 1.0f / (1.0f + exp(- (&kernelContext_0)->logits_0[tok_0 * n_expert_0 + e_0]));
        (*(&kernelContext_0)->sh_prob_0)[e_0] = p_0;
        (*(&kernelContext_0)->sh_score_0)[e_0] = p_0 + (&kernelContext_0)->bias_0[e_0];

#line 70
        e_0 = e_0 + 32U;

#line 70
    }


    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 82
    if(_S2 != 0U)
    {

#line 83
        return;
    }

#line 92
    thread array<uint, int(16)> chosen_0;
    thread array<float, int(16)> unnorm_w_0;

#line 92
    uint i_0 = 0U;
    for(;;)
    {

#line 93
        if(i_0 < 16U)
        {
        }
        else
        {

#line 93
            break;
        }

#line 94
        chosen_0[i_0] = 4294967295U;
        unnorm_w_0[i_0] = 0.0f;

#line 93
        i_0 = i_0 + 1U;

#line 93
    }

#line 93
    uint s_0 = 0U;

#line 93
    float sum_0 = 0.0f;

#line 98
    for(;;)
    {

#line 98
        if(s_0 < _S1)
        {
        }
        else
        {

#line 98
            break;
        }

#line 98
        bool have_0 = false;

#line 98
        float best_score_0 = 0.0f;

#line 98
        uint best_0 = 0U;

#line 98
        e_0 = 0U;

#line 103
        for(;;)
        {

#line 103
            if(e_0 < n_expert_0)
            {
            }
            else
            {

#line 103
                break;
            }

#line 103
            bool used_0 = false;

#line 103
            uint t_0 = 0U;

            for(;;)
            {

#line 105
                if(t_0 < s_0)
                {
                }
                else
                {

#line 105
                    break;
                }

#line 106
                if((chosen_0[t_0]) == e_0)
                {

#line 106
                    used_0 = true;

#line 106
                }

#line 105
                t_0 = t_0 + 1U;

#line 105
            }

#line 110
            if(used_0)
            {

#line 111
                e_0 = e_0 + 1U;

#line 103
                continue;
            }

#line 113
            float score_0 = (*(&kernelContext_0)->sh_score_0)[e_0];

#line 113
            bool _S3;
            if(!have_0)
            {

#line 114
                _S3 = true;

#line 114
            }
            else
            {

#line 114
                _S3 = score_0 > best_score_0;

#line 114
            }

#line 114
            uint best_1;

#line 114
            float best_score_1;

#line 114
            bool have_1;

#line 114
            if(_S3)
            {

#line 114
                have_1 = true;

#line 114
                best_score_1 = score_0;

#line 114
                best_1 = e_0;

#line 114
            }
            else
            {

#line 114
                have_1 = have_0;

#line 114
                best_score_1 = best_score_0;

#line 114
                best_1 = best_0;

#line 114
            }

#line 114
            have_0 = have_1;

#line 114
            best_score_0 = best_score_1;

#line 114
            best_0 = best_1;

#line 103
            e_0 = e_0 + 1U;

#line 103
        }

#line 121
        chosen_0[s_0] = best_0;
        float w_0 = (*(&kernelContext_0)->sh_prob_0)[best_0];
        unnorm_w_0[s_0] = w_0;
        uint _S4 = tok_0 * _S1 + s_0;

#line 123
        *((&kernelContext_0)->sel_expert_0+_S4) = best_0;
        float sum_1 = sum_0 + w_0;

#line 98
        s_0 = s_0 + 1U;

#line 98
        sum_0 = sum_1;

#line 98
    }

#line 130
    float inv_denom_0 = 1.0f / max(sum_0, 0.00006103515625f);

#line 130
    s_0 = 0U;
    for(;;)
    {

#line 131
        if(s_0 < _S1)
        {
        }
        else
        {

#line 131
            break;
        }

#line 132
        uint _S6 = tok_0 * _S1 + s_0;

#line 132
        *((&kernelContext_0)->sel_weight_0+_S6) = unnorm_w_0[s_0] * inv_denom_0;

#line 131
        s_0 = s_0 + 1U;

#line 131
    }


    return;
}

