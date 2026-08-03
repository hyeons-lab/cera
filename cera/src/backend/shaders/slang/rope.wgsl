@binding(2) @group(0) var<storage, read> params_0 : array<u32>;

@binding(0) @group(0) var<storage, read_write> q_0 : array<f32>;

@binding(1) @group(0) var<storage, read_write> k_0 : array<f32>;

@binding(3) @group(0) var<storage, read> freq_factors_0 : array<f32>;

fn rotate_pair_0( x0_0 : f32,  x1_0 : f32,  angle_0 : f32) -> vec2<f32>
{
    var cos_a_0 : f32 = cos(angle_0);
    var sin_a_0 : f32 = sin(angle_0);
    return vec2<f32>(x0_0 * cos_a_0 - x1_0 * sin_a_0, x0_0 * sin_a_0 + x1_0 * cos_a_0);
}

@compute
@workgroup_size(256, 1, 1)
fn rope(@builtin(global_invocation_id) gid_0 : vec3<u32>)
{
    var idx_0 : u32 = gid_0.x;
    var pos_0 : u32 = params_0[i32(0)];
    var n_kv_heads_0 : u32 = params_0[i32(2)];
    var head_dim_0 : u32 = params_0[i32(3)];
    var freq_base_0 : f32 = (bitcast<f32>((params_0[i32(4)])));
    var rope_type_0 : u32 = params_0[i32(5)];
    var has_freq_factors_0 : u32 = params_0[i32(6)];
    var half_dim_0 : u32 = head_dim_0 / u32(2);
    var angle_1 : f32;
    var i0_0 : u32;
    var i1_0 : u32;
    if(idx_0 < (params_0[i32(1)] * half_dim_0))
    {
        var head_0 : u32 = idx_0 / half_dim_0;
        var d_0 : u32 = idx_0 % half_dim_0;
        var angle_2 : f32 = f32(pos_0) * pow(freq_base_0, -2.0f * f32(d_0) / f32(head_dim_0));
        if(has_freq_factors_0 == u32(1))
        {
            angle_1 = angle_2 / freq_factors_0[d_0];
        }
        else
        {
            angle_1 = angle_2;
        }
        if(rope_type_0 == u32(0))
        {
            var _S1 : u32 = head_0 * head_dim_0 + d_0;
            var _S2 : u32 = _S1 + half_dim_0;
            i0_0 = _S1;
            i1_0 = _S2;
        }
        else
        {
            var _S3 : u32 = head_0 * head_dim_0 + u32(2) * d_0;
            var _S4 : u32 = _S3 + u32(1);
            i0_0 = _S3;
            i1_0 = _S4;
        }
        var r_0 : vec2<f32> = rotate_pair_0(q_0[i0_0], q_0[i1_0], angle_1);
        q_0[i0_0] = r_0.x;
        q_0[i1_0] = r_0.y;
    }
    if(idx_0 < (n_kv_heads_0 * half_dim_0))
    {
        var head_1 : u32 = idx_0 / half_dim_0;
        var d_1 : u32 = idx_0 % half_dim_0;
        var angle_3 : f32 = f32(pos_0) * pow(freq_base_0, -2.0f * f32(d_1) / f32(head_dim_0));
        if(has_freq_factors_0 == u32(1))
        {
            angle_1 = angle_3 / freq_factors_0[d_1];
        }
        else
        {
            angle_1 = angle_3;
        }
        if(rope_type_0 == u32(0))
        {
            var _S5 : u32 = head_1 * head_dim_0 + d_1;
            var _S6 : u32 = _S5 + half_dim_0;
            i0_0 = _S5;
            i1_0 = _S6;
        }
        else
        {
            var _S7 : u32 = head_1 * head_dim_0 + u32(2) * d_1;
            var _S8 : u32 = _S7 + u32(1);
            i0_0 = _S7;
            i1_0 = _S8;
        }
        var r_1 : vec2<f32> = rotate_pair_0(k_0[i0_0], k_0[i1_0], angle_1);
        k_0[i0_0] = r_1.x;
        k_0[i1_0] = r_1.y;
    }
    return;
}

