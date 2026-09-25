#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 24 "cera/src/backend/shaders/slang/kv_append.slang"
uint to_f16_bits_0(float v_0)
{

#line 25
    uint b_0 = (as_type<uint>((v_0)));
    uint s_0 = (b_0 >> 16U) & 32768U;
    int e_0 = int((b_0 >> 23U) & 255U) - int(112);
    uint m_0 = b_0 & 8388607U;

#line 28
    uint h_0;

    if(e_0 >= int(31))
    {

#line 30
        h_0 = 31743U;

#line 30
    }
    else
    {

#line 32
        if(e_0 <= int(0))
        {

#line 32
            h_0 = 0U;

#line 32
        }
        else
        {

#line 32
            h_0 = (uint(e_0) << 10U) | ((m_0 + 4096U) >> 13U);

#line 32
        }

#line 30
    }

#line 39
    return s_0 | h_0;
}


#line 52
struct KernelContext_0
{
    packed_uint4 device* par_buf_0;
    uint device* dst_buf_0;
    float device* src_buf_0;
};


#line 44
[[kernel]] void kv_append(uint3 gid_0 [[thread_position_in_grid]], packed_uint4 device* par_buf_1 [[buffer(2)]], uint device* dst_buf_1 [[buffer(1)]], float device* src_buf_1 [[buffer(0)]])
{

#line 44
    thread KernelContext_0 kernelContext_0;

#line 44
    (&kernelContext_0)->par_buf_0 = par_buf_1;

#line 44
    (&kernelContext_0)->dst_buf_0 = dst_buf_1;

#line 44
    (&kernelContext_0)->src_buf_0 = src_buf_1;
    uint w_0 = gid_0.x;
    uint off_0 = (uint4(*(par_buf_1+int(0))) ).x;

    if(w_0 >= (((uint4(*(par_buf_1+int(0))) ).y) >> 1U))
    {

#line 49
        return;
    }
    uint i0_0 = w_0 << 1U;
    *((&kernelContext_0)->dst_buf_0+(off_0 + w_0)) = (to_f16_bits_0((&kernelContext_0)->src_buf_0[i0_0])) | ((to_f16_bits_0((&kernelContext_0)->src_buf_0[i0_0 + 1U])) << 16U);
    return;
}

