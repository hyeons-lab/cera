#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 27 "cera/src/backend/shaders/slang/ffn_swiglu_q4_0.slang"
uint get_wid_0(uint3 wid_0)
{

#line 28
    return wid_0.x + wid_0.y * 65535U;
}

float block_scale_0(uint device* w_0, uint blk_byte_0)
{

#line 32
    uint word_0 = w_0[blk_byte_0 >> 2U];

#line 32
    uint scale_bits_0;
    if((blk_byte_0 & 2U) != 0U)
    {

#line 33
        scale_bits_0 = word_0 >> 16U;

#line 33
    }
    else
    {

#line 33
        scale_bits_0 = word_0 & 65535U;

#line 33
    }
    return (as_type<half>((ushort)((scale_bits_0))));
}

uint q_pair_0(uint device* w_1, uint qs_byte_0)
{

#line 38
    uint word_1 = w_1[qs_byte_0 >> 2U];

#line 38
    uint _S1;
    if((qs_byte_0 & 2U) != 0U)
    {

#line 39
        _S1 = word_1 >> 16U;

#line 39
    }
    else
    {

#line 39
        _S1 = word_1 & 65535U;

#line 39
    }

#line 39
    return _S1;
}


#line 164
struct KernelContext_0
{
    packed_uint4 device* params_0;
    float device* x_0;
    uint device* w_gate_0;
    uint device* w_up_0;
    float device* y_0;
    array<float, int(512)> threadgroup* x_stage_0;
    array<float, int(128)> threadgroup* partials_gate_0;
    array<float, int(128)> threadgroup* partials_up_0;
};


#line 44
[[kernel]] void ffn_swiglu_q4_0(uint3 lid_0 [[thread_position_in_threadgroup]], uint3 wid_1 [[threadgroup_position_in_grid]], packed_uint4 device* params_1 [[buffer(4)]], float device* x_1 [[buffer(2)]], uint device* w_gate_1 [[buffer(0)]], uint device* w_up_1 [[buffer(1)]], float device* y_1 [[buffer(3)]])
{
    uint i_0;

#line 46
    thread KernelContext_0 kernelContext_0;

#line 46
    (&kernelContext_0)->params_0 = params_1;

#line 46
    (&kernelContext_0)->x_0 = x_1;

#line 46
    (&kernelContext_0)->w_gate_0 = w_gate_1;

#line 46
    (&kernelContext_0)->w_up_0 = w_up_1;

#line 46
    (&kernelContext_0)->y_0 = y_1;

#line 46
    threadgroup array<float, int(512)> x_stage_1;

#line 46
    (&kernelContext_0)->x_stage_0 = &x_stage_1;

#line 46
    threadgroup array<float, int(128)> partials_gate_1;

#line 46
    (&kernelContext_0)->partials_gate_0 = &partials_gate_1;

#line 46
    threadgroup array<float, int(128)> partials_up_1;

#line 46
    (&kernelContext_0)->partials_up_0 = &partials_up_1;

    uint _S2 = (uint4(*(params_1+int(0))) ).x;
    uint k_0 = (uint4(*(params_1+int(0))) ).y;
    uint nb_0 = k_0 / 32U;
    uint _S3 = nb_0 * 18U;
    uint _S4 = get_wid_0(wid_1) * 4U;
    uint tid_0 = lid_0.x;

    uint _S5 = tid_0 / 2U;
    uint _S6 = (tid_0 & 1U) * 8U;

    thread array<float, int(4)> sum_gate_0;
    thread array<float, int(4)> sum_up_0;

#line 59
    uint r_0 = 0U;

    for(;;)
    {

#line 61
        if(r_0 < 4U)
        {
        }
        else
        {

#line 61
            break;
        }

#line 62
        sum_gate_0[r_0] = 0.0f;
        sum_up_0[r_0] = 0.0f;

#line 61
        r_0 = r_0 + 1U;

#line 61
    }

#line 61
    uint chunk_b_0 = 0U;

#line 66
    for(;;)
    {

#line 66
        if(chunk_b_0 < nb_0)
        {
        }
        else
        {

#line 66
            break;
        }

#line 67
        uint chunk_k_start_0 = chunk_b_0 * 32U;

        uint _S7 = min(512U, k_0 - chunk_k_start_0) / 4U;

#line 69
        i_0 = tid_0;
        for(;;)
        {

#line 70
            if(i_0 < _S7)
            {
            }
            else
            {

#line 70
                break;
            }

#line 71
            uint _S8 = i_0 * 4U;

#line 71
            uint base_idx_0 = chunk_k_start_0 + _S8;
            (*(&kernelContext_0)->x_stage_0)[_S8] = (&kernelContext_0)->x_0[base_idx_0];
            (*(&kernelContext_0)->x_stage_0)[_S8 + 1U] = (&kernelContext_0)->x_0[base_idx_0 + 1U];
            (*(&kernelContext_0)->x_stage_0)[_S8 + 2U] = (&kernelContext_0)->x_0[base_idx_0 + 2U];
            (*(&kernelContext_0)->x_stage_0)[_S8 + 3U] = (&kernelContext_0)->x_0[base_idx_0 + 3U];

#line 70
            i_0 = i_0 + 32U;

#line 70
        }

#line 77
        threadgroup_barrier(mem_flags::mem_threadgroup);

#line 77
        uint ib_local_0 = _S5;


        for(;;)
        {

#line 80
            bool _S9;

#line 80
            if(ib_local_0 < 16U)
            {

#line 80
                _S9 = (chunk_b_0 + ib_local_0) < nb_0;

#line 80
            }
            else
            {

#line 80
                _S9 = false;

#line 80
            }

#line 80
            if(_S9)
            {
            }
            else
            {

#line 80
                break;
            }

#line 81
            uint _S10 = chunk_b_0 + ib_local_0;
            uint yb_stage_off_0 = ib_local_0 * 32U + _S6;

            float a0_0 = (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0];

            float a2_0 = (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 2U];

            float a4_0 = (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 4U];

            float a6_0 = (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 6U];

#line 102
            float _S11 = (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 1U] / 256.0f;

            float _S12 = (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 3U] / 256.0f;

            float _S13 = (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 5U] / 256.0f;

            float _S14 = (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 7U] / 256.0f;
            float _S15 = (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 16U] / 16.0f;
            float _S16 = (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 17U] / 4096.0f;
            float _S17 = (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 18U] / 16.0f;
            float _S18 = (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 19U] / 4096.0f;
            float _S19 = (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 20U] / 16.0f;
            float _S20 = (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 21U] / 4096.0f;
            float _S21 = (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 22U] / 16.0f;
            float _S22 = (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 23U] / 4096.0f;



            float _S23 = (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0] + (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 1U] + ((*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 2U] + (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 3U]) + ((*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 4U] + (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 5U]) + ((*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 6U] + (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 7U]) + ((*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 16U] + (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 17U] + ((*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 18U] + (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 19U]) + ((*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 20U] + (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 21U]) + ((*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 22U] + (*(&kernelContext_0)->x_stage_0)[yb_stage_off_0 + 23U]));

#line 120
            r_0 = 0U;


            for(;;)
            {

#line 123
                if(r_0 < 4U)
                {
                }
                else
                {

#line 123
                    break;
                }

#line 124
                uint _S24 = _S4 + r_0;

#line 124
                if(_S24 >= _S2)
                {

#line 124
                    r_0 = r_0 + 1U;

#line 123
                    continue;
                }
                uint blk_byte_1 = _S24 * _S3 + _S10 * 18U;
                uint qs_byte_1 = blk_byte_1 + 2U + _S6;



                uint gq0_0 = q_pair_0((&kernelContext_0)->w_gate_0, qs_byte_1);
                uint _S25 = qs_byte_1 + 2U;

#line 131
                uint gq1_0 = q_pair_0((&kernelContext_0)->w_gate_0, _S25);
                uint _S26 = qs_byte_1 + 4U;

#line 132
                uint gq2_0 = q_pair_0((&kernelContext_0)->w_gate_0, _S26);
                uint _S27 = qs_byte_1 + 6U;

#line 133
                uint gq3_0 = q_pair_0((&kernelContext_0)->w_gate_0, _S27);

#line 140
                float _S28 = _S23 * -8.0f;

#line 140
                sum_gate_0[r_0] = sum_gate_0[r_0] + block_scale_0((&kernelContext_0)->w_gate_0, blk_byte_1) * (_S28 + (a0_0 * float(gq0_0 & 15U) + a2_0 * float(gq1_0 & 15U) + a4_0 * float(gq2_0 & 15U) + a6_0 * float(gq3_0 & 15U)) + (_S11 * float(gq0_0 & 3840U) + _S12 * float(gq1_0 & 3840U) + _S13 * float(gq2_0 & 3840U) + _S14 * float(gq3_0 & 3840U)) + (_S15 * float(gq0_0 & 240U) + _S17 * float(gq1_0 & 240U) + _S19 * float(gq2_0 & 240U) + _S21 * float(gq3_0 & 240U)) + (_S16 * float(gq0_0 & 61440U) + _S18 * float(gq1_0 & 61440U) + _S20 * float(gq2_0 & 61440U) + _S22 * float(gq3_0 & 61440U)));



                uint uq0_0 = q_pair_0((&kernelContext_0)->w_up_0, qs_byte_1);
                uint uq1_0 = q_pair_0((&kernelContext_0)->w_up_0, _S25);
                uint uq2_0 = q_pair_0((&kernelContext_0)->w_up_0, _S26);
                uint uq3_0 = q_pair_0((&kernelContext_0)->w_up_0, _S27);

#line 154
                sum_up_0[r_0] = sum_up_0[r_0] + block_scale_0((&kernelContext_0)->w_up_0, blk_byte_1) * (_S28 + (a0_0 * float(uq0_0 & 15U) + a2_0 * float(uq1_0 & 15U) + a4_0 * float(uq2_0 & 15U) + a6_0 * float(uq3_0 & 15U)) + (_S11 * float(uq0_0 & 3840U) + _S12 * float(uq1_0 & 3840U) + _S13 * float(uq2_0 & 3840U) + _S14 * float(uq3_0 & 3840U)) + (_S15 * float(uq0_0 & 240U) + _S17 * float(uq1_0 & 240U) + _S19 * float(uq2_0 & 240U) + _S21 * float(uq3_0 & 240U)) + (_S16 * float(uq0_0 & 61440U) + _S18 * float(uq1_0 & 61440U) + _S20 * float(uq2_0 & 61440U) + _S22 * float(uq3_0 & 61440U)));

#line 123
                r_0 = r_0 + 1U;

#line 123
            }

#line 123
            ib_local_0 = ib_local_0 + 16U;

#line 80
        }

#line 159
        threadgroup_barrier(mem_flags::mem_threadgroup);

#line 66
        chunk_b_0 = chunk_b_0 + 16U;

#line 66
    }

#line 66
    r_0 = 0U;

#line 163
    for(;;)
    {

#line 163
        if(r_0 < 4U)
        {
        }
        else
        {

#line 163
            break;
        }

#line 164
        uint _S29 = r_0 * 32U + tid_0;

#line 164
        (*(&kernelContext_0)->partials_gate_0)[_S29] = sum_gate_0[r_0];
        (*(&kernelContext_0)->partials_up_0)[_S29] = sum_up_0[r_0];

#line 163
        r_0 = r_0 + 1U;

#line 163
    }



    threadgroup_barrier(mem_flags::mem_threadgroup);

#line 167
    i_0 = 16U;

    for(;;)
    {

#line 169
        if(i_0 > 0U)
        {
        }
        else
        {

#line 169
            break;
        }

#line 170
        if(tid_0 < i_0)
        {

#line 170
            r_0 = 0U;

            for(;;)
            {

#line 172
                if(r_0 < 4U)
                {
                }
                else
                {

#line 172
                    break;
                }

#line 173
                uint idx_0 = r_0 * 32U + tid_0;
                uint _S30 = idx_0 + i_0;

#line 174
                (*(&kernelContext_0)->partials_gate_0)[idx_0] = (*(&kernelContext_0)->partials_gate_0)[idx_0] + (*(&kernelContext_0)->partials_gate_0)[_S30];
                (*(&kernelContext_0)->partials_up_0)[idx_0] = (*(&kernelContext_0)->partials_up_0)[idx_0] + (*(&kernelContext_0)->partials_up_0)[_S30];

#line 172
                r_0 = r_0 + 1U;

#line 172
            }

#line 170
        }

#line 178
        threadgroup_barrier(mem_flags::mem_threadgroup);

#line 169
        i_0 = i_0 >> 1U;

#line 169
    }

#line 181
    if(tid_0 == 0U)
    {

#line 181
        r_0 = 0U;

        for(;;)
        {

#line 183
            if(r_0 < 4U)
            {
            }
            else
            {

#line 183
                break;
            }

#line 184
            uint _S31 = _S4 + r_0;

#line 184
            if(_S31 < _S2)
            {

#line 185
                uint _S32 = r_0 * 32U;



                *((&kernelContext_0)->y_0+_S31) = (*(&kernelContext_0)->partials_gate_0)[_S32] * (1.0f / (1.0f + exp2(- clamp((*(&kernelContext_0)->partials_gate_0)[_S32], -80.0f, 80.0f) * 1.4426950216293335f))) * (*(&kernelContext_0)->partials_up_0)[_S32];

#line 184
            }

#line 183
            r_0 = r_0 + 1U;

#line 183
        }

#line 181
    }

#line 193
    return;
}

