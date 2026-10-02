#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 71 "cera/src/backend/shaders/slang/deltanet_recurrence.slang"
struct KernelContext_0
{
    uint device* par_buf_0;
    float device* alpha_buf_0;
    float device* beta_buf_0;
    float device* v_buf_0;
    float device* state_buf_0;
    float device* k_buf_0;
    float device* q_buf_0;
    float device* out_buf_0;
};


#line 39
[[kernel]] void deltanet_recurrence(uint3 gid_0 [[threadgroup_position_in_grid]], uint3 lid_0 [[thread_position_in_threadgroup]], uint device* par_buf_1 [[buffer(7)]], float device* alpha_buf_1 [[buffer(3)]], float device* beta_buf_1 [[buffer(4)]], float device* v_buf_1 [[buffer(2)]], float device* state_buf_1 [[buffer(5)]], float device* k_buf_1 [[buffer(1)]], float device* q_buf_1 [[buffer(0)]], float device* out_buf_1 [[buffer(6)]])
{
    float s_val_0;

#line 41
    thread KernelContext_0 kernelContext_0;

#line 41
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 41
    (&kernelContext_0)->alpha_buf_0 = alpha_buf_1;

#line 41
    (&kernelContext_0)->beta_buf_0 = beta_buf_1;

#line 41
    (&kernelContext_0)->v_buf_0 = v_buf_1;

#line 41
    (&kernelContext_0)->state_buf_0 = state_buf_1;

#line 41
    (&kernelContext_0)->k_buf_0 = k_buf_1;

#line 41
    (&kernelContext_0)->q_buf_0 = q_buf_1;

#line 41
    (&kernelContext_0)->out_buf_0 = out_buf_1;

    uint h_0 = gid_0.x;
    uint j_0 = gid_0.y * 128U + lid_0.x;

    uint num_v_heads_0 = par_buf_1[int(0)];
    uint num_k_heads_0 = par_buf_1[int(1)];
    uint head_k_dim_0 = par_buf_1[int(2)];
    uint head_v_dim_0 = par_buf_1[int(3)];

#line 49
    bool _S1;

    if(h_0 >= num_v_heads_0)
    {

#line 51
        _S1 = true;

#line 51
    }
    else
    {

#line 51
        _S1 = j_0 >= head_v_dim_0;

#line 51
    }

#line 51
    if(_S1)
    {

#line 52
        return;
    }

    uint _S2 = num_v_heads_0 / max(1U, num_k_heads_0);
    uint _S3 = h_0 / max(1U, _S2);

    float _S4 = (&kernelContext_0)->alpha_buf_0[h_0];
    float b_0 = (&kernelContext_0)->beta_buf_0[h_0];
    uint _S5 = h_0 * head_v_dim_0 + j_0;

#line 60
    float vj_0 = (&kernelContext_0)->v_buf_0[_S5];

    uint _S6 = h_0 * head_k_dim_0 * head_v_dim_0;
    uint _S7 = min(_S3, num_k_heads_0 - 1U) * head_k_dim_0;


    thread array<float, int(128)> s_col_0;

#line 66
    uint i_0 = 0U;

#line 66
    float sk_j_0 = 0.0f;



    for(;;)
    {

#line 70
        if(i_0 < head_k_dim_0)
        {
        }
        else
        {

#line 70
            break;
        }

#line 71
        float s_val_1 = *((&kernelContext_0)->state_buf_0+(_S6 + i_0 * head_v_dim_0 + j_0));
        if(_S4 <= 0.0f)
        {

#line 72
            s_val_0 = 0.0f;

#line 72
        }
        else
        {

#line 74
            if((abs(_S4 - 1.0f)) > 1.00000001168609742e-07f)
            {

#line 74
                s_val_0 = s_val_1 * _S4;

#line 74
            }
            else
            {

#line 74
                s_val_0 = s_val_1;

#line 74
            }

#line 72
        }

#line 77
        if(i_0 < 128U)
        {

#line 78
            s_col_0[i_0] = s_val_0;

#line 77
        }


        float sk_j_1 = sk_j_0 + s_val_0 * (&kernelContext_0)->k_buf_0[_S7 + i_0];

#line 70
        i_0 = i_0 + 1U;

#line 70
        sk_j_0 = sk_j_1;

#line 70
    }

#line 84
    float _S8 = b_0 * (vj_0 - sk_j_0);

#line 84
    i_0 = 0U;

#line 84
    float oj_0 = 0.0f;

#line 90
    for(;;)
    {

#line 90
        if(i_0 < head_k_dim_0)
        {
        }
        else
        {

#line 90
            break;
        }

#line 91
        uint idx_0 = _S6 + i_0 * head_v_dim_0 + j_0;
        if(i_0 < 128U)
        {

#line 92
            s_val_0 = s_col_0[i_0];

#line 92
        }
        else
        {

#line 92
            float _S9 = *((&kernelContext_0)->state_buf_0+idx_0);

#line 92
            if(_S4 <= 0.0f)
            {

#line 92
                sk_j_0 = 0.0f;

#line 92
            }
            else
            {

#line 92
                sk_j_0 = _S4;

#line 92
            }

#line 92
            s_val_0 = _S9 * sk_j_0;

#line 92
        }
        uint _S10 = _S7 + i_0;

#line 93
        float updated_0 = s_val_0 + (&kernelContext_0)->k_buf_0[_S10] * _S8;
        *((&kernelContext_0)->state_buf_0+idx_0) = updated_0;
        float oj_1 = oj_0 + updated_0 * (&kernelContext_0)->q_buf_0[_S10];

#line 90
        i_0 = i_0 + 1U;

#line 90
        oj_0 = oj_1;

#line 90
    }

#line 98
    *((&kernelContext_0)->out_buf_0+_S5) = oj_0;
    return;
}

