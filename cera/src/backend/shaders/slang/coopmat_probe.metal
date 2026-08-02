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
simdgroup_matrix<float, int(8), int(8)> linalg_coopMatMulAdd_0(simdgroup_matrix<half, int(8), int(8)> matA_0, simdgroup_matrix<half, int(8), int(8)> matB_0, simdgroup_matrix<float, int(8), int(8)> matC_0)
{

#line 30057
    simdgroup_matrix<float, int(8), int(8)> _S1;
    simdgroup_multiply_accumulate(_S1, matA_0, matB_0, matC_0);

#line 30057
    return _S1;
}


#line 44 "cera/src/backend/shaders/slang/coopmat_probe.slang"
struct KernelContext_0
{
    half device* a_buf_0;
    half device* b_buf_0;
    float device* c_buf_0;
};


#line 44
[[kernel]] void coopmat_probe(uint3 tid_0 [[thread_position_in_threadgroup]], half device* a_buf_1 [[buffer(0)]], half device* b_buf_1 [[buffer(1)]], float device* c_buf_1 [[buffer(2)]])
{

#line 44
    thread KernelContext_0 kernelContext_0;

#line 44
    (&kernelContext_0)->a_buf_0 = a_buf_1;

#line 44
    (&kernelContext_0)->b_buf_0 = b_buf_1;

#line 44
    (&kernelContext_0)->c_buf_0 = c_buf_1;

#line 53
    simdgroup_matrix<half, int(8), int(8)> a_0 = (_slang_simdgroup_load<simdgroup_matrix<half, int(8), int(8)>>((const device half*)((a_buf_1)) + (0U), (ulong)(8U)));
    simdgroup_matrix<half, int(8), int(8)> b_0 = (_slang_simdgroup_load<simdgroup_matrix<half, int(8), int(8)>>((const device half*)((b_buf_1)) + (0U), (ulong)(8U)));
    thread simdgroup_matrix<float, int(8), int(8)> _S2;

#line 55
    _slang_simdgroup_fill((&_S2), (0.0f));
    simdgroup_matrix<float, int(8), int(8)> c_0 = linalg_coopMatMulAdd_0(a_0, b_0, _S2);
    linalg_CoopMat_Store_0(c_0, (&kernelContext_0)->c_buf_0, 0U, 8U);

#line 78
    return;
}

