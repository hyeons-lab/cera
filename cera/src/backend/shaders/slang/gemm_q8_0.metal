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


#line 80 "cera/src/backend/shaders/slang/gemm_q8_0.slang"
struct GemmParams_0
{
    uint m_0;
    uint k_0;
    uint n_0;
    uint x_stride_0;
    uint y_stride_0;
    uint _pad_0;
};


#line 177
struct KernelContext_0
{
    GemmParams_0 device* params_0;
    uint device* src0_0;
    float device* src1_0;
    float device* dst_0;
    array<half, int(2048)> threadgroup* sa_0;
    array<float, int(1024)> threadgroup* sb_0;
};


#line 176
[[kernel]] void gemm_q8_0(uint3 sv_groupthreadid_0 [[thread_position_in_threadgroup]], uint3 gid_0 [[threadgroup_position_in_grid]], GemmParams_0 device* params_1 [[buffer(3)]], uint device* src0_1 [[buffer(0)]], float device* src1_1 [[buffer(1)]], float device* dst_1 [[buffer(2)]])
{

#line 176
    uint ik_0;

#line 176
    uint lsma_0;

#line 176
    thread KernelContext_0 kernelContext_0;

#line 176
    (&kernelContext_0)->params_0 = params_1;

#line 176
    (&kernelContext_0)->src0_0 = src0_1;

#line 176
    (&kernelContext_0)->src1_0 = src1_1;

#line 176
    (&kernelContext_0)->dst_0 = dst_1;

#line 176
    threadgroup array<half, int(2048)> sa_1;

#line 176
    (&kernelContext_0)->sa_0 = &sa_1;

#line 176
    threadgroup array<float, int(1024)> sb_1;

#line 176
    (&kernelContext_0)->sb_0 = &sb_1;

#line 176
    uint sv_groupindex_0 = (sv_groupthreadid_0[int(2)] + sv_groupthreadid_0[int(1)]) * 128U + sv_groupthreadid_0[int(0)];
    GemmParams_0 _S2 = params_1[int(0)];
    GemmParams_0 _S3 = params_1[int(0)];
    GemmParams_0 _S4 = params_1[int(0)];

    GemmParams_0 _S5 = params_1[int(0)];

#line 186
    uint r0_0 = gid_0.y;
    uint r1_0 = gid_0.x;

#line 198
    uint sgitg_0 = sv_groupindex_0 >> 5U;

    uint _S6 = r0_0 * 64U;

#line 200
    uint _S7 = min(_S2.m_0 - _S6, 64U);
    uint _S8 = r1_0 * 32U;

#line 201
    uint _S9 = min(_S4.n_0 - _S8, 32U);

#line 214
    uint _S10 = sv_groupindex_0 % 2U;

    uint _S11 = _S3.k_0 / 32U * 34U * (_S6 + min(sv_groupindex_0 / 2U, _S7 - 1U));

    uint _S12 = sv_groupindex_0 % 4U;

#line 218
    uint _S13 = params_1[int(0)].x_stride_0 * (_S8 + min(sv_groupindex_0 / 4U, _S9 - 1U)) + 8U * _S12;

    thread array<simdgroup_matrix<float, int(8), int(8)>, int(8)> mc_0;


    thread simdgroup_matrix<float, int(8), int(8)> _S14;

#line 223
    _slang_simdgroup_fill((&_S14), (0.0f));

#line 223
    mc_0[0U] = _S14;

#line 223
    thread simdgroup_matrix<float, int(8), int(8)> _S15;

#line 223
    _slang_simdgroup_fill((&_S15), (0.0f));

#line 223
    mc_0[1U] = _S15;

#line 223
    thread simdgroup_matrix<float, int(8), int(8)> _S16;

#line 223
    _slang_simdgroup_fill((&_S16), (0.0f));

#line 223
    mc_0[2U] = _S16;

#line 223
    thread simdgroup_matrix<float, int(8), int(8)> _S17;

#line 223
    _slang_simdgroup_fill((&_S17), (0.0f));

#line 223
    mc_0[3U] = _S17;

#line 223
    thread simdgroup_matrix<float, int(8), int(8)> _S18;

#line 223
    _slang_simdgroup_fill((&_S18), (0.0f));

#line 223
    mc_0[4U] = _S18;

#line 223
    thread simdgroup_matrix<float, int(8), int(8)> _S19;

#line 223
    _slang_simdgroup_fill((&_S19), (0.0f));

#line 223
    mc_0[5U] = _S19;

#line 223
    thread simdgroup_matrix<float, int(8), int(8)> _S20;

#line 223
    _slang_simdgroup_fill((&_S20), (0.0f));

#line 223
    mc_0[6U] = _S20;

#line 223
    thread simdgroup_matrix<float, int(8), int(8)> _S21;

#line 223
    _slang_simdgroup_fill((&_S21), (0.0f));

#line 223
    mc_0[7U] = _S21;

#line 223
    uint loop_k_0 = 0U;

#line 223
    uint x_byte_0 = _S11;

#line 223
    uint y_off_0 = _S13;


    for(;;)
    {

#line 226
        if(loop_k_0 < (_S3.k_0))
        {
        }
        else
        {

#line 226
            break;
        }

        half _S22 = ((*(const device half*)((const device char*)((&kernelContext_0)->src0_0) + (x_byte_0))));

#line 229
        float _S23 = float(_S22);
        thread array<half, int(16)> temp_a_0;
        uint _S24 = x_byte_0 + 2U + 16U * _S10;


        int4 q_0 = (int4(*(const device packed_char4*)((const device char*)((&kernelContext_0)->src0_0) + (_S24))));
        temp_a_0[0U] = half(float(q_0.x) * _S23);
        temp_a_0[1U] = half(float(q_0.y) * _S23);
        temp_a_0[2U] = half(float(q_0.z) * _S23);
        temp_a_0[3U] = half(float(q_0.w) * _S23);

#line 234
        int4 q_1 = (int4(*(const device packed_char4*)((const device char*)((&kernelContext_0)->src0_0) + (_S24 + 4U))));
        temp_a_0[4U] = half(float(q_1.x) * _S23);
        temp_a_0[5U] = half(float(q_1.y) * _S23);
        temp_a_0[6U] = half(float(q_1.z) * _S23);
        temp_a_0[7U] = half(float(q_1.w) * _S23);

#line 234
        int4 q_2 = (int4(*(const device packed_char4*)((const device char*)((&kernelContext_0)->src0_0) + (_S24 + 8U))));
        temp_a_0[8U] = half(float(q_2.x) * _S23);
        temp_a_0[9U] = half(float(q_2.y) * _S23);
        temp_a_0[10U] = half(float(q_2.z) * _S23);
        temp_a_0[11U] = half(float(q_2.w) * _S23);

#line 234
        int4 q_3 = (int4(*(const device packed_char4*)((const device char*)((&kernelContext_0)->src0_0) + (_S24 + 12U))));
        temp_a_0[12U] = half(float(q_3.x) * _S23);
        temp_a_0[13U] = half(float(q_3.y) * _S23);
        temp_a_0[14U] = half(float(q_3.z) * _S23);
        temp_a_0[15U] = half(float(q_3.w) * _S23);


        threadgroup_barrier(mem_flags::mem_threadgroup);

#line 249
        uint _S25 = _S10 * 16U;


        (*(&kernelContext_0)->sa_0)[64U * (sv_groupindex_0 / 2U / 8U + _S25) + sv_groupindex_0 / 2U % 8U] = temp_a_0[0U];

#line 252
        (*(&kernelContext_0)->sa_0)[64U * (sv_groupindex_0 / 2U / 8U + _S25) + sv_groupindex_0 / 2U % 8U + 8U] = temp_a_0[1U];

#line 252
        (*(&kernelContext_0)->sa_0)[64U * (sv_groupindex_0 / 2U / 8U + _S25) + sv_groupindex_0 / 2U % 8U + 16U] = temp_a_0[2U];

#line 252
        (*(&kernelContext_0)->sa_0)[64U * (sv_groupindex_0 / 2U / 8U + _S25) + sv_groupindex_0 / 2U % 8U + 24U] = temp_a_0[3U];

#line 252
        (*(&kernelContext_0)->sa_0)[64U * (sv_groupindex_0 / 2U / 8U + _S25) + sv_groupindex_0 / 2U % 8U + 32U] = temp_a_0[4U];

#line 252
        (*(&kernelContext_0)->sa_0)[64U * (sv_groupindex_0 / 2U / 8U + _S25) + sv_groupindex_0 / 2U % 8U + 40U] = temp_a_0[5U];

#line 252
        (*(&kernelContext_0)->sa_0)[64U * (sv_groupindex_0 / 2U / 8U + _S25) + sv_groupindex_0 / 2U % 8U + 48U] = temp_a_0[6U];

#line 252
        (*(&kernelContext_0)->sa_0)[64U * (sv_groupindex_0 / 2U / 8U + _S25) + sv_groupindex_0 / 2U % 8U + 56U] = temp_a_0[7U];

#line 252
        (*(&kernelContext_0)->sa_0)[64U * (sv_groupindex_0 / 2U / 8U + _S25 + 8U) + sv_groupindex_0 / 2U % 8U] = temp_a_0[8U];

#line 252
        (*(&kernelContext_0)->sa_0)[64U * (sv_groupindex_0 / 2U / 8U + _S25 + 8U) + sv_groupindex_0 / 2U % 8U + 8U] = temp_a_0[9U];

#line 252
        (*(&kernelContext_0)->sa_0)[64U * (sv_groupindex_0 / 2U / 8U + _S25 + 8U) + sv_groupindex_0 / 2U % 8U + 16U] = temp_a_0[10U];

#line 252
        (*(&kernelContext_0)->sa_0)[64U * (sv_groupindex_0 / 2U / 8U + _S25 + 8U) + sv_groupindex_0 / 2U % 8U + 24U] = temp_a_0[11U];

#line 252
        (*(&kernelContext_0)->sa_0)[64U * (sv_groupindex_0 / 2U / 8U + _S25 + 8U) + sv_groupindex_0 / 2U % 8U + 32U] = temp_a_0[12U];

#line 252
        (*(&kernelContext_0)->sa_0)[64U * (sv_groupindex_0 / 2U / 8U + _S25 + 8U) + sv_groupindex_0 / 2U % 8U + 40U] = temp_a_0[13U];

#line 252
        (*(&kernelContext_0)->sa_0)[64U * (sv_groupindex_0 / 2U / 8U + _S25 + 8U) + sv_groupindex_0 / 2U % 8U + 48U] = temp_a_0[14U];

#line 252
        (*(&kernelContext_0)->sa_0)[64U * (sv_groupindex_0 / 2U / 8U + _S25 + 8U) + sv_groupindex_0 / 2U % 8U + 56U] = temp_a_0[15U];

#line 259
        (*(threadgroup float2x4*)(&(*(&kernelContext_0)->sb_0)[(256U * _S12 + 8U * (sv_groupindex_0 / 4U))]) = *(const device float2x4*)(&((&kernelContext_0)->src1_0)[(y_off_0)]));

        uint x_byte_1 = x_byte_0 + 34U;
        uint y_off_1 = y_off_0 + 32U;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint _S26 = 256U * (sgitg_0 % 2U);
        uint _S27 = 128U * (sgitg_0 / 2U);

#line 267
        ik_0 = 0U;

#line 267
        lsma_0 = _S26;

#line 267
        uint lsmb_0 = _S27;

#line 280
        for(;;)
        {

#line 280
            if(ik_0 < 4U)
            {
            }
            else
            {

#line 280
                break;
            }

#line 281
            thread array<simdgroup_matrix<half, int(8), int(8)>, int(4)> ma_0;
            thread array<simdgroup_matrix<float, int(8), int(8)>, int(2)> mb_0;



            simdgroup_matrix<half, int(8), int(8)> _S28 = (_slang_simdgroup_load<simdgroup_matrix<half, int(8), int(8)>>(&((*(((&kernelContext_0)->sa_0)))[0]) + (lsma_0), (ulong)(8U)));

#line 286
            ma_0[0U] = _S28;

#line 286
            simdgroup_matrix<half, int(8), int(8)> _S29 = (_slang_simdgroup_load<simdgroup_matrix<half, int(8), int(8)>>(&((*(((&kernelContext_0)->sa_0)))[0]) + (lsma_0 + 64U), (ulong)(8U)));

#line 286
            ma_0[1U] = _S29;

#line 286
            simdgroup_matrix<half, int(8), int(8)> _S30 = (_slang_simdgroup_load<simdgroup_matrix<half, int(8), int(8)>>(&((*(((&kernelContext_0)->sa_0)))[0]) + (lsma_0 + 128U), (ulong)(8U)));

#line 286
            ma_0[2U] = _S30;

#line 286
            simdgroup_matrix<half, int(8), int(8)> _S31 = (_slang_simdgroup_load<simdgroup_matrix<half, int(8), int(8)>>(&((*(((&kernelContext_0)->sa_0)))[0]) + (lsma_0 + 192U), (ulong)(8U)));

#line 286
            ma_0[3U] = _S31;



            simdgroup_matrix<float, int(8), int(8)> _S32 = (_slang_simdgroup_load<simdgroup_matrix<float, int(8), int(8)>>(&((*(((&kernelContext_0)->sb_0)))[0]) + (lsmb_0), (ulong)(8U)));

#line 290
            mb_0[0U] = _S32;

#line 290
            simdgroup_matrix<float, int(8), int(8)> _S33 = (_slang_simdgroup_load<simdgroup_matrix<float, int(8), int(8)>>(&((*(((&kernelContext_0)->sb_0)))[0]) + (lsmb_0 + 64U), (ulong)(8U)));

#line 290
            mb_0[1U] = _S33;

#line 299
            simdgroup_matrix<float, int(8), int(8)> _S34 = linalg_coopMatMulAdd_0(mb_0[0U], ma_0[0U], mc_0[0U]);

#line 299
            mc_0[0U] = _S34;

#line 299
            simdgroup_matrix<float, int(8), int(8)> _S35 = linalg_coopMatMulAdd_0(mb_0[0U], ma_0[1U], mc_0[1U]);

#line 299
            mc_0[1U] = _S35;

#line 299
            simdgroup_matrix<float, int(8), int(8)> _S36 = linalg_coopMatMulAdd_0(mb_0[0U], ma_0[2U], mc_0[2U]);

#line 299
            mc_0[2U] = _S36;

#line 299
            simdgroup_matrix<float, int(8), int(8)> _S37 = linalg_coopMatMulAdd_0(mb_0[0U], ma_0[3U], mc_0[3U]);

#line 299
            mc_0[3U] = _S37;

#line 299
            simdgroup_matrix<float, int(8), int(8)> _S38 = linalg_coopMatMulAdd_0(mb_0[1U], ma_0[0U], mc_0[4U]);

#line 299
            mc_0[4U] = _S38;

#line 299
            simdgroup_matrix<float, int(8), int(8)> _S39 = linalg_coopMatMulAdd_0(mb_0[1U], ma_0[1U], mc_0[5U]);

#line 299
            mc_0[5U] = _S39;

#line 299
            simdgroup_matrix<float, int(8), int(8)> _S40 = linalg_coopMatMulAdd_0(mb_0[1U], ma_0[2U], mc_0[6U]);

#line 299
            mc_0[6U] = _S40;

#line 299
            simdgroup_matrix<float, int(8), int(8)> _S41 = linalg_coopMatMulAdd_0(mb_0[1U], ma_0[3U], mc_0[7U]);

#line 299
            mc_0[7U] = _S41;


            uint lsma_1 = lsma_0 + 512U;
            uint lsmb_1 = lsmb_0 + 256U;

#line 280
            ik_0 = ik_0 + 1U;

#line 280
            lsma_0 = lsma_1;

#line 280
            lsmb_0 = lsmb_1;

#line 280
        }

#line 226
        loop_k_0 = loop_k_0 + 32U;

#line 226
        x_byte_0 = x_byte_1;

#line 226
        y_off_0 = y_off_1;

#line 226
    }

#line 226
    bool _S42;

#line 307
    if(((r0_0 + 1U) * 64U) <= (_S2.m_0))
    {

#line 307
        _S42 = ((r1_0 + 1U) * 32U) <= (_S4.n_0);

#line 307
    }
    else
    {

#line 307
        _S42 = false;

#line 307
    }

#line 307
    if(_S42)
    {

        uint _S43 = 64U * r0_0 + 32U * (sgitg_0 & 1U) + (32U * r1_0 + 16U * (sgitg_0 >> 1U)) * _S5.y_stride_0;


        linalg_CoopMat_Store_0(mc_0[0U], (&kernelContext_0)->dst_0, _S43, _S5.y_stride_0);
        uint _S44 = _S43 + 8U;

#line 313
        linalg_CoopMat_Store_0(mc_0[1U], (&kernelContext_0)->dst_0, _S44, _S5.y_stride_0);
        uint _S45 = _S43 + 16U;

#line 313
        linalg_CoopMat_Store_0(mc_0[2U], (&kernelContext_0)->dst_0, _S45, _S5.y_stride_0);
        uint _S46 = _S43 + 24U;

#line 313
        linalg_CoopMat_Store_0(mc_0[3U], (&kernelContext_0)->dst_0, _S46, _S5.y_stride_0);
        uint _S47 = 8U * _S5.y_stride_0;

#line 313
        linalg_CoopMat_Store_0(mc_0[4U], (&kernelContext_0)->dst_0, _S43 + _S47, _S5.y_stride_0);

#line 313
        linalg_CoopMat_Store_0(mc_0[5U], (&kernelContext_0)->dst_0, _S44 + _S47, _S5.y_stride_0);

#line 313
        linalg_CoopMat_Store_0(mc_0[6U], (&kernelContext_0)->dst_0, _S45 + _S47, _S5.y_stride_0);

#line 313
        linalg_CoopMat_Store_0(mc_0[7U], (&kernelContext_0)->dst_0, _S46 + _S47, _S5.y_stride_0);

#line 307
    }
    else
    {

#line 307
        ik_0 = 0U;

#line 320
        for(;;)
        {

#line 320
            if(ik_0 < 2U)
            {
            }
            else
            {

#line 320
                break;
            }

#line 321
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if((sgitg_0 >> 1U) == ik_0)
            {

#line 324
                uint _S48 = 32U * (sgitg_0 & 1U);


                simdgroup_store((mc_0[0U]), &((*(((&kernelContext_0)->sb_0)))[0]) + (_S48), (ulong)(64U));
                uint _S49 = _S48 + 8U;

#line 327
                simdgroup_store((mc_0[1U]), &((*(((&kernelContext_0)->sb_0)))[0]) + (_S49), (ulong)(64U));
                uint _S50 = _S48 + 16U;

#line 327
                simdgroup_store((mc_0[2U]), &((*(((&kernelContext_0)->sb_0)))[0]) + (_S50), (ulong)(64U));
                uint _S51 = _S48 + 24U;

#line 327
                simdgroup_store((mc_0[3U]), &((*(((&kernelContext_0)->sb_0)))[0]) + (_S51), (ulong)(64U));

#line 327
                simdgroup_store((mc_0[4U]), &((*(((&kernelContext_0)->sb_0)))[0]) + (_S48 + 512U), (ulong)(64U));

#line 327
                simdgroup_store((mc_0[5U]), &((*(((&kernelContext_0)->sb_0)))[0]) + (_S49 + 512U), (ulong)(64U));

#line 327
                simdgroup_store((mc_0[6U]), &((*(((&kernelContext_0)->sb_0)))[0]) + (_S50 + 512U), (ulong)(64U));

#line 327
                simdgroup_store((mc_0[7U]), &((*(((&kernelContext_0)->sb_0)))[0]) + (_S51 + 512U), (ulong)(64U));

#line 323
            }

#line 332
            threadgroup_barrier(mem_flags::mem_threadgroup);

#line 332
            lsma_0 = sv_groupindex_0;

#line 337
            for(;;)
            {

#line 337
                if(lsma_0 < 1024U)
                {
                }
                else
                {

#line 337
                    break;
                }

#line 338
                uint j_0 = lsma_0 / 64U;
                uint r_0 = lsma_0 % 64U;
                uint gj_0 = ik_0 * 16U + j_0;
                if(gj_0 < _S9)
                {

#line 341
                    _S42 = r_0 < _S7;

#line 341
                }
                else
                {

#line 341
                    _S42 = false;

#line 341
                }

#line 341
                if(_S42)
                {
                    *((&kernelContext_0)->dst_0+(_S6 + r_0 + (_S8 + gj_0) * _S5.y_stride_0)) = (*(&kernelContext_0)->sb_0)[j_0 * 64U + r_0];

#line 341
                }

#line 337
                lsma_0 = lsma_0 + 128U;

#line 337
            }

#line 320
            ik_0 = ik_0 + 1U;

#line 320
        }

#line 307
    }

#line 378
    return;
}

