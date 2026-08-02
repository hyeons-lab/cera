#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#include <metal_simdgroup_matrix>
template<typename T, int Cols, int Rows, typename V>
void _slang_simdgroup_fill(thread simdgroup_matrix<T, Cols, Rows>* dest, V val) {
    *dest = make_filled_simdgroup_matrix<T, Cols, Rows>(T(val));
}
template<typename Matrix, typename T>
Matrix _slang_simdgroup_load(const device T* src, ulong elements_per_row) {
    Matrix result;
    simdgroup_load(result, src, elements_per_row);
    return result;
}
template<typename Matrix, typename T>
Matrix _slang_simdgroup_load_transpose(const device T* src, ulong elements_per_row) {
    Matrix result;
    simdgroup_load(result, src, elements_per_row, ulong2(0), true);
    return result;
}
template<typename Matrix, typename T>
Matrix _slang_simdgroup_load(const threadgroup T* src, ulong elements_per_row) {
    Matrix result;
    simdgroup_load(result, src, elements_per_row);
    return result;
}
template<typename Matrix, typename T>
Matrix _slang_simdgroup_load_transpose(const threadgroup T* src, ulong elements_per_row) {
    Matrix result;
    simdgroup_load(result, src, elements_per_row, ulong2(0), true);
    return result;
}


#line 28665 "hlsl.meta.slang"
void linalg_CoopMat_Store_0(simdgroup_matrix<float, int(8), int(8)> this_0, float device* buffer_0, uint element_0, uint stride_0)
{



    simdgroup_store((this_0), (device float*)((buffer_0)) + (element_0), (ulong)(stride_0));
    return;
}


#line 30042
simdgroup_matrix<float, int(8), int(8)> linalg_coopMatMulAdd_0(simdgroup_matrix<float, int(8), int(8)> matA_0, simdgroup_matrix<half, int(8), int(8)> matB_0, simdgroup_matrix<float, int(8), int(8)> matC_0)
{

#line 30057
    simdgroup_matrix<float, int(8), int(8)> _S1;
    simdgroup_multiply_accumulate(_S1, matA_0, matB_0, matC_0);

#line 30057
    return _S1;
}


#line 57 "cera/src/backend/shaders/slang/gemm_q8_0.slang"
struct GemmParams_0
{
    uint m_0;
    uint k_0;
    uint n_0;
    uint x_stride_0;
    uint y_stride_0;
    uint _pad_0;
};


#line 193
struct KernelContext_0
{
    GemmParams_0 device* params_0;
    uint device* src0_0;
    float device* src1_0;
    float device* dst_0;
    array<half, int(2048)> threadgroup* sa_0;
    array<float, int(1024)> threadgroup* sb_0;
};


#line 108
float load_f16_0(uint byte_off_0, KernelContext_0 thread* kernelContext_0)
{

#line 109
    uint w_0 = kernelContext_0->src0_0[byte_off_0 >> 2U];

#line 109
    uint h_0;
    if((byte_off_0 & 2U) != 0U)
    {

#line 110
        h_0 = w_0 >> 16U;

#line 110
    }
    else
    {

#line 110
        h_0 = w_0 & 65535U;

#line 110
    }
    return (as_type<half>((ushort)((h_0))));
}


#line 97
uint load_byte_0(uint byte_off_1, KernelContext_0 thread* kernelContext_1)
{
    return (kernelContext_1->src0_0[byte_off_1 >> 2U] >> ((byte_off_1 & 3U) * 8U)) & 255U;
}


int load_i8_0(uint byte_off_2, KernelContext_0 thread* kernelContext_2)
{

#line 103
    uint _S2 = load_byte_0(byte_off_2, kernelContext_2);
    return int(_S2 ^ 128U) - int(128);
}


#line 116
[[kernel]] void gemm_q8_0(uint3 sv_groupthreadid_0 [[thread_position_in_threadgroup]], uint3 gid_0 [[threadgroup_position_in_grid]], GemmParams_0 device* params_1 [[buffer(3)]], uint device* src0_1 [[buffer(0)]], float device* src1_1 [[buffer(1)]], float device* dst_1 [[buffer(2)]])
{

#line 116
    uint e_0;

#line 116
    uint e_1;

#line 116
    thread KernelContext_0 kernelContext_3;

#line 116
    (&kernelContext_3)->params_0 = params_1;

#line 116
    (&kernelContext_3)->src0_0 = src0_1;

#line 116
    (&kernelContext_3)->src1_0 = src1_1;

#line 116
    (&kernelContext_3)->dst_0 = dst_1;

#line 116
    threadgroup array<half, int(2048)> sa_1;

#line 116
    (&kernelContext_3)->sa_0 = &sa_1;

#line 116
    threadgroup array<float, int(1024)> sb_1;

#line 116
    (&kernelContext_3)->sb_0 = &sb_1;

#line 116
    uint sv_groupindex_0 = (sv_groupthreadid_0[int(2)] + sv_groupthreadid_0[int(1)]) * 128U + sv_groupthreadid_0[int(0)];
    GemmParams_0 _S3 = params_1[int(0)];
    GemmParams_0 _S4 = params_1[int(0)];
    GemmParams_0 _S5 = params_1[int(0)];

    GemmParams_0 _S6 = params_1[int(0)];

#line 126
    uint r0_0 = gid_0.y;
    uint r1_0 = gid_0.x;

#line 138
    uint sgitg_0 = sv_groupindex_0 >> 5U;

    uint _S7 = r0_0 * 64U;

#line 140
    uint _S8 = min(_S3.m_0 - _S7, 64U);
    uint _S9 = r1_0 * 32U;

#line 141
    uint _S10 = min(_S5.n_0 - _S9, 32U);

#line 154
    uint _S11 = sv_groupindex_0 % 2U;

    uint _S12 = _S4.k_0 / 32U * 34U * (_S7 + min(sv_groupindex_0 / 2U, _S8 - 1U));

    uint _S13 = sv_groupindex_0 % 4U;

#line 158
    uint _S14 = params_1[int(0)].x_stride_0 * (_S9 + min(sv_groupindex_0 / 4U, _S10 - 1U)) + 8U * _S13;

    thread array<simdgroup_matrix<float, int(8), int(8)>, int(8)> mc_0;

#line 160
    uint i_0 = 0U;
    for(;;)
    {

#line 161
        if(i_0 < 8U)
        {
        }
        else
        {

#line 161
            break;
        }

#line 162
        thread simdgroup_matrix<float, int(8), int(8)> _S15;

#line 162
        _slang_simdgroup_fill((&_S15), (0.0f));

#line 162
        mc_0[i_0] = _S15;

#line 161
        i_0 = i_0 + 1U;

#line 161
    }

#line 161
    uint loop_k_0 = 0U;

#line 161
    uint x_byte_0 = _S12;

#line 161
    uint y_off_0 = _S14;



    for(;;)
    {

#line 165
        if(loop_k_0 < (_S4.k_0))
        {
        }
        else
        {

#line 165
            break;
        }

#line 165
        float _S16 = load_f16_0(x_byte_0, &kernelContext_3);



        thread array<float, int(16)> temp_a_0;

#line 169
        e_0 = 0U;
        for(;;)
        {

#line 170
            if(e_0 < 16U)
            {
            }
            else
            {

#line 170
                break;
            }

#line 170
            int _S17 = load_i8_0(x_byte_0 + 2U + e_0 + 16U * _S11, &kernelContext_3);
            temp_a_0[e_0] = float(_S17) * _S16;

#line 170
            e_0 = e_0 + 1U;

#line 170
        }



        threadgroup_barrier(mem_flags::mem_threadgroup);

#line 174
        i_0 = 0U;

#line 179
        for(;;)
        {

#line 179
            if(i_0 < 16U)
            {
            }
            else
            {

#line 179
                break;
            }



            (*(&kernelContext_3)->sa_0)[64U * (sv_groupindex_0 / 2U / 8U + _S11 * 16U + i_0 / 8U * 8U) + sv_groupindex_0 / 2U % 8U + (i_0 & 7U) * 8U] = half(temp_a_0[i_0]);

#line 179
            i_0 = i_0 + 1U;

#line 179
        }

#line 191
        uint _S18 = 256U * _S13 + 8U * (sv_groupindex_0 / 4U);

#line 191
        e_1 = 0U;
        for(;;)
        {

#line 192
            if(e_1 < 8U)
            {
            }
            else
            {

#line 192
                break;
            }

#line 193
            (*(&kernelContext_3)->sb_0)[_S18 + e_1] = (&kernelContext_3)->src1_0[y_off_0 + e_1];

#line 192
            e_1 = e_1 + 1U;

#line 192
        }



        uint x_byte_1 = x_byte_0 + 34U;
        uint y_off_1 = y_off_0 + 32U;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint _S19 = 256U * (sgitg_0 % 2U);
        uint _S20 = 128U * (sgitg_0 / 2U);

#line 202
        uint ik_0 = 0U;

#line 202
        uint lsma_0 = _S19;

#line 202
        uint lsmb_0 = _S20;

        for(;;)
        {

#line 204
            if(ik_0 < 4U)
            {
            }
            else
            {

#line 204
                break;
            }

#line 205
            thread array<simdgroup_matrix<half, int(8), int(8)>, int(4)> ma_0;
            thread array<simdgroup_matrix<float, int(8), int(8)>, int(2)> mb_0;

#line 206
            uint i_1 = 0U;

            for(;;)
            {

#line 208
                if(i_1 < 4U)
                {
                }
                else
                {

#line 208
                    break;
                }

#line 209
                simdgroup_matrix<half, int(8), int(8)> _S21 = (_slang_simdgroup_load<simdgroup_matrix<half, int(8), int(8)>>(&((*(((&kernelContext_3)->sa_0)))[0]) + (lsma_0 + 64U * i_1), (ulong)(8U)));

#line 209
                ma_0[i_1] = _S21;

#line 208
                i_1 = i_1 + 1U;

#line 208
            }

#line 208
            uint i_2 = 0U;


            for(;;)
            {

#line 211
                if(i_2 < 2U)
                {
                }
                else
                {

#line 211
                    break;
                }

#line 212
                simdgroup_matrix<float, int(8), int(8)> _S22 = (_slang_simdgroup_load<simdgroup_matrix<float, int(8), int(8)>>(&((*(((&kernelContext_3)->sb_0)))[0]) + (lsmb_0 + 64U * i_2), (ulong)(8U)));

#line 212
                mb_0[i_2] = _S22;

#line 211
                i_2 = i_2 + 1U;

#line 211
            }

#line 211
            uint i_3 = 0U;

#line 219
            for(;;)
            {

#line 219
                if(i_3 < 8U)
                {
                }
                else
                {

#line 219
                    break;
                }

#line 220
                simdgroup_matrix<float, int(8), int(8)> _S23 = linalg_coopMatMulAdd_0(mb_0[i_3 / 4U], ma_0[i_3 % 4U], mc_0[i_3]);

#line 220
                mc_0[i_3] = _S23;

#line 219
                i_3 = i_3 + 1U;

#line 219
            }



            uint lsma_1 = lsma_0 + 512U;
            uint lsmb_1 = lsmb_0 + 256U;

#line 204
            ik_0 = ik_0 + 1U;

#line 204
            lsma_0 = lsma_1;

#line 204
            lsmb_0 = lsmb_1;

#line 204
        }

#line 165
        loop_k_0 = loop_k_0 + 32U;

#line 165
        x_byte_0 = x_byte_1;

#line 165
        y_off_0 = y_off_1;

#line 165
    }

#line 165
    bool _S24;

#line 228
    if(((r0_0 + 1U) * 64U) <= (_S3.m_0))
    {

#line 228
        _S24 = ((r1_0 + 1U) * 32U) <= (_S5.n_0);

#line 228
    }
    else
    {

#line 228
        _S24 = false;

#line 228
    }

#line 228
    if(_S24)
    {

        uint _S25 = 64U * r0_0 + 32U * (sgitg_0 & 1U) + (32U * r1_0 + 16U * (sgitg_0 >> 1U)) * _S6.y_stride_0;

#line 231
        i_0 = 0U;
        for(;;)
        {

#line 232
            if(i_0 < 8U)
            {
            }
            else
            {

#line 232
                break;
            }

#line 233
            linalg_CoopMat_Store_0(mc_0[i_0], (&kernelContext_3)->dst_0, _S25 + 8U * (i_0 % 4U) + 8U * _S6.y_stride_0 * (i_0 / 4U), _S6.y_stride_0);

#line 232
            i_0 = i_0 + 1U;

#line 232
        }

#line 228
    }
    else
    {

#line 228
        e_0 = 0U;

#line 240
        for(;;)
        {

#line 240
            if(e_0 < 2U)
            {
            }
            else
            {

#line 240
                break;
            }

#line 241
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if((sgitg_0 >> 1U) == e_0)
            {

#line 244
                uint _S26 = 32U * (sgitg_0 & 1U);

#line 244
                i_0 = 0U;
                for(;;)
                {

#line 245
                    if(i_0 < 8U)
                    {
                    }
                    else
                    {

#line 245
                        break;
                    }

#line 246
                    simdgroup_store((mc_0[i_0]), &((*(((&kernelContext_3)->sb_0)))[0]) + (_S26 + 8U * (i_0 % 4U) + 512U * (i_0 / 4U)), (ulong)(64U));

#line 245
                    i_0 = i_0 + 1U;

#line 245
                }

#line 243
            }

#line 251
            threadgroup_barrier(mem_flags::mem_threadgroup);

#line 251
            e_1 = sv_groupindex_0;

#line 256
            for(;;)
            {

#line 256
                if(e_1 < 1024U)
                {
                }
                else
                {

#line 256
                    break;
                }

#line 257
                uint j_0 = e_1 / 64U;
                uint r_0 = e_1 % 64U;
                uint gj_0 = e_0 * 16U + j_0;
                if(gj_0 < _S10)
                {

#line 260
                    _S24 = r_0 < _S8;

#line 260
                }
                else
                {

#line 260
                    _S24 = false;

#line 260
                }

#line 260
                if(_S24)
                {
                    *((&kernelContext_3)->dst_0+(_S7 + r_0 + (_S9 + gj_0) * _S6.y_stride_0)) = (*(&kernelContext_3)->sb_0)[j_0 * 64U + r_0];

#line 260
                }

#line 256
                e_1 = e_1 + 128U;

#line 256
            }

#line 240
            e_0 = e_0 + 1U;

#line 240
        }

#line 228
    }

#line 297
    return;
}

