@binding(7) @group(0) var<storage, read> par_buf_0 : array<u32>;

@binding(3) @group(0) var<storage, read> alpha_buf_0 : array<f32>;

@binding(4) @group(0) var<storage, read> beta_buf_0 : array<f32>;

@binding(2) @group(0) var<storage, read> v_buf_0 : array<f32>;

@binding(5) @group(0) var<storage, read_write> state_buf_0 : array<f32>;

@binding(1) @group(0) var<storage, read> k_buf_0 : array<f32>;

@binding(0) @group(0) var<storage, read> q_buf_0 : array<f32>;

@binding(6) @group(0) var<storage, read_write> out_buf_0 : array<f32>;

@compute
@workgroup_size(128, 1, 1)
fn deltanet_recurrence(@builtin(workgroup_id) gid_0 : vec3<u32>, @builtin(local_invocation_id) lid_0 : vec3<u32>)
{
    var s_val_0 : f32;
    var h_0 : u32 = gid_0.x;
    var j_0 : u32 = gid_0.y * u32(128) + lid_0.x;
    var num_v_heads_0 : u32 = par_buf_0[i32(0)];
    var num_k_heads_0 : u32 = par_buf_0[i32(1)];
    var head_k_dim_0 : u32 = par_buf_0[i32(2)];
    var head_v_dim_0 : u32 = par_buf_0[i32(3)];
    var _S1 : bool;
    if(h_0 >= num_v_heads_0)
    {
        _S1 = true;
    }
    else
    {
        _S1 = j_0 >= head_v_dim_0;
    }
    if(_S1)
    {
        return;
    }
    var _S2 : u32 = num_v_heads_0 / max(u32(1), num_k_heads_0);
    var _S3 : u32 = h_0 / max(u32(1), _S2);
    var _S4 : f32 = alpha_buf_0[h_0];
    var b_0 : f32 = beta_buf_0[h_0];
    var _S5 : u32 = h_0 * head_v_dim_0 + j_0;
    var vj_0 : f32 = v_buf_0[_S5];
    var _S6 : u32 = h_0 * head_k_dim_0 * head_v_dim_0;
    var _S7 : u32 = min(_S3, num_k_heads_0 - u32(1)) * head_k_dim_0;
    var s_col_0 : array<f32, i32(128)>;
    var i_0 : u32 = u32(0);
    var sk_j_0 : f32 = 0.0f;
    loop
    {
        if(i_0 < head_k_dim_0)
        {
        }
        else
        {
            break;
        }
        var s_val_1 : f32 = state_buf_0[_S6 + i_0 * head_v_dim_0 + j_0];
        if(_S4 <= 0.0f)
        {
            s_val_0 = 0.0f;
        }
        else
        {
            if((abs(_S4 - 1.0f)) > 1.00000001168609742e-07f)
            {
                s_val_0 = s_val_1 * _S4;
            }
            else
            {
                s_val_0 = s_val_1;
            }
        }
        if(i_0 < u32(128))
        {
            s_col_0[i_0] = s_val_0;
        }
        var sk_j_1 : f32 = sk_j_0 + s_val_0 * k_buf_0[_S7 + i_0];
        i_0 = i_0 + u32(1);
        sk_j_0 = sk_j_1;
    }
    var _S8 : f32 = b_0 * (vj_0 - sk_j_0);
    i_0 = u32(0);
    var oj_0 : f32 = 0.0f;
    loop
    {
        if(i_0 < head_k_dim_0)
        {
        }
        else
        {
            break;
        }
        var idx_0 : u32 = _S6 + i_0 * head_v_dim_0 + j_0;
        if(i_0 < u32(128))
        {
            s_val_0 = s_col_0[i_0];
        }
        else
        {
            var _S9 : f32 = state_buf_0[idx_0];
            if(_S4 <= 0.0f)
            {
                sk_j_0 = 0.0f;
            }
            else
            {
                sk_j_0 = _S4;
            }
            s_val_0 = _S9 * sk_j_0;
        }
        var _S10 : u32 = _S7 + i_0;
        var updated_0 : f32 = s_val_0 + k_buf_0[_S10] * _S8;
        state_buf_0[idx_0] = updated_0;
        var oj_1 : f32 = oj_0 + updated_0 * q_buf_0[_S10];
        i_0 = i_0 + u32(1);
        oj_0 = oj_1;
    }
    out_buf_0[_S5] = oj_0;
    return;
}

