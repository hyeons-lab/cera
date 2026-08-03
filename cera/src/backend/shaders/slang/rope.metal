#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;

#line 30 "cera/src/backend/shaders/slang/rope.slang"
float2 rotate_pair_0(float x0_0, float x1_0, float angle_0)
{

#line 31
    float cos_a_0 = cos(angle_0);
    float sin_a_0 = sin(angle_0);
    return float2(x0_0 * cos_a_0 - x1_0 * sin_a_0, x0_0 * sin_a_0 + x1_0 * cos_a_0);
}


#line 63
struct KernelContext_0
{
    uint device* params_0;
    float device* q_0;
    float device* k_0;
};


#line 38
[[kernel]] void rope(uint3 gid_0 [[thread_position_in_grid]], uint device* params_1 [[buffer(2)]], float device* q_1 [[buffer(0)]], float device* k_1 [[buffer(1)]])
{

#line 38
    thread KernelContext_0 kernelContext_0;

#line 38
    (&kernelContext_0)->params_0 = params_1;

#line 38
    (&kernelContext_0)->q_0 = q_1;

#line 38
    (&kernelContext_0)->k_0 = k_1;
    uint idx_0 = gid_0.x;

#line 45
    uint pos_0 = params_1[int(0)];

    uint n_kv_heads_0 = params_1[int(2)];
    uint head_dim_0 = params_1[int(3)];
    float freq_base_0 = (as_type<float>((params_1[int(4)])));
    uint half_dim_0 = head_dim_0 / 2U;

#line 56
    if(idx_0 < (params_1[int(1)] * half_dim_0))
    {

#line 57
        uint head_0 = idx_0 / half_dim_0;
        uint d_0 = idx_0 % half_dim_0;


        uint i0_0 = head_0 * head_dim_0 + d_0;
        uint i1_0 = i0_0 + half_dim_0;
        float2 r_0 = rotate_pair_0(*((&kernelContext_0)->q_0+i0_0), *((&kernelContext_0)->q_0+i1_0), float(pos_0) * (1.0f / pow(freq_base_0, float(2U * d_0) / float(head_dim_0))));
        *((&kernelContext_0)->q_0+i0_0) = r_0.x;
        *((&kernelContext_0)->q_0+i1_0) = r_0.y;

#line 56
    }

#line 69
    if(idx_0 < (n_kv_heads_0 * half_dim_0))
    {

#line 70
        uint head_1 = idx_0 / half_dim_0;
        uint d_1 = idx_0 % half_dim_0;


        uint i0_1 = head_1 * head_dim_0 + d_1;
        uint i1_1 = i0_1 + half_dim_0;
        float2 r_1 = rotate_pair_0(*((&kernelContext_0)->k_0+i0_1), *((&kernelContext_0)->k_0+i1_1), float(pos_0) * (1.0f / pow(freq_base_0, float(2U * d_1) / float(head_dim_0))));
        *((&kernelContext_0)->k_0+i0_1) = r_1.x;
        *((&kernelContext_0)->k_0+i1_1) = r_1.y;

#line 69
    }

#line 141
    return;
}

