@binding(7) @group(0) var<storage, read> par_buf_0 : array<vec4<u32>>;

@binding(0) @group(0) var<storage, read> q_buf_0 : array<f32>;

@binding(4) @group(0) var<storage, read> bu_buf_0 : array<f32>;

@binding(5) @group(0) var<storage, read> bv_buf_0 : array<f32>;

@binding(1) @group(0) var<storage, read> k_buf_0 : array<f32>;

@binding(3) @group(0) var<storage, read> p_buf_0 : array<f32>;

@binding(2) @group(0) var<storage, read> v_buf_0 : array<f32>;

@binding(6) @group(0) var<storage, read_write> out_buf_0 : array<f32>;

var<workgroup> qu_0 : array<f32, i32(128)>;

var<workgroup> qv_0 : array<f32, i32(128)>;

var<workgroup> scores_0 : array<f32, i32(1024)>;

var<workgroup> scratch_0 : array<f32, i32(256)>;

fn block_max_0( tid_0 : u32,  v_0 : f32) -> f32
{
    scratch_0[tid_0] = v_0;
    workgroupBarrier();
    var s_0 : u32 = u32(128);
    for(;;)
    {
        if(s_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_0 < s_0)
        {
            scratch_0[tid_0] = max(scratch_0[tid_0], scratch_0[tid_0 + s_0]);
        }
        workgroupBarrier();
        s_0 = (s_0 >> (u32(1)));
    }
    var _S1 : f32 = scratch_0[i32(0)];
    return _S1;
}

fn block_sum_0( tid_1 : u32,  v_1 : f32) -> f32
{
    scratch_0[tid_1] = v_1;
    workgroupBarrier();
    var s_1 : u32 = u32(128);
    for(;;)
    {
        if(s_1 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_1 < s_1)
        {
            scratch_0[tid_1] = scratch_0[tid_1] + scratch_0[tid_1 + s_1];
        }
        workgroupBarrier();
        s_1 = (s_1 >> (u32(1)));
    }
    var _S2 : f32 = scratch_0[i32(0)];
    return _S2;
}

@compute
@workgroup_size(256, 1, 1)
fn audio_xl_attention(@builtin(local_invocation_id) lid_0 : vec3<u32>, @builtin(workgroup_id) wid_0 : vec3<u32>)
{
    var ac_0 : f32;
    var bd_0 : f32;
    var tid_2 : u32 = lid_0.x;
    var tokens_0 : u32 = par_buf_0[i32(0)].x;
    var head_dim_0 : u32 = par_buf_0[i32(0)].z;
    var _S3 : f32 = (bitcast<f32>((par_buf_0[i32(0)].w)));
    var q_idx_0 : u32 = wid_0.x;
    var dim_0 : u32 = par_buf_0[i32(0)].y * head_dim_0;
    var head_base_0 : u32 = wid_0.y * head_dim_0;
    var _S4 : u32 = q_idx_0 * dim_0 + head_base_0;
    var d_0 : u32 = tid_2;
    for(;;)
    {
        if(d_0 < head_dim_0)
        {
        }
        else
        {
            break;
        }
        var qd_0 : f32 = q_buf_0[_S4 + d_0];
        var _S5 : u32 = head_base_0 + d_0;
        qu_0[d_0] = qd_0 + bu_buf_0[_S5];
        qv_0[d_0] = qd_0 + bv_buf_0[_S5];
        d_0 = d_0 + u32(256);
    }
    workgroupBarrier();
    var _S6 : u32 = tokens_0 - u32(1);
    var key_0 : u32 = tid_2;
    for(;;)
    {
        if(key_0 < tokens_0)
        {
        }
        else
        {
            break;
        }
        var _S7 : u32 = key_0 * dim_0 + head_base_0;
        var _S8 : u32 = (_S6 + key_0 - q_idx_0) * dim_0 + head_base_0;
        d_0 = u32(0);
        ac_0 = 0.0f;
        bd_0 = 0.0f;
        for(;;)
        {
            if(d_0 < head_dim_0)
            {
            }
            else
            {
                break;
            }
            var ac_1 : f32 = ac_0 + qu_0[d_0] * k_buf_0[_S7 + d_0];
            var bd_1 : f32 = bd_0 + qv_0[d_0] * p_buf_0[_S8 + d_0];
            d_0 = d_0 + u32(1);
            ac_0 = ac_1;
            bd_0 = bd_1;
        }
        scores_0[key_0] = (ac_0 + bd_0) * _S3;
        key_0 = key_0 + u32(256);
    }
    workgroupBarrier();
    ac_0 = -3.4028234663852886e+38f;
    key_0 = tid_2;
    for(;;)
    {
        if(key_0 < tokens_0)
        {
        }
        else
        {
            break;
        }
        var _S9 : f32 = max(ac_0, scores_0[key_0]);
        var key_1 : u32 = key_0 + u32(256);
        ac_0 = _S9;
        key_0 = key_1;
    }
    var _S10 : f32 = block_max_0(tid_2, ac_0);
    key_0 = tid_2;
    bd_0 = 0.0f;
    for(;;)
    {
        if(key_0 < tokens_0)
        {
        }
        else
        {
            break;
        }
        var e_0 : f32 = exp(scores_0[key_0] - _S10);
        scores_0[key_0] = e_0;
        var partial_0 : f32 = bd_0 + e_0;
        key_0 = key_0 + u32(256);
        bd_0 = partial_0;
    }
    workgroupBarrier();
    var _S11 : f32 = block_sum_0(tid_2, bd_0);
    var _S12 : f32 = 1.0f / _S11;
    d_0 = tid_2;
    for(;;)
    {
        if(d_0 < head_dim_0)
        {
        }
        else
        {
            break;
        }
        key_0 = u32(0);
        var acc_0 : f32 = 0.0f;
        for(;;)
        {
            if(key_0 < tokens_0)
            {
            }
            else
            {
                break;
            }
            var acc_1 : f32 = acc_0 + scores_0[key_0] * v_buf_0[key_0 * dim_0 + head_base_0 + d_0];
            key_0 = key_0 + u32(1);
            acc_0 = acc_1;
        }
        out_buf_0[_S4 + d_0] = acc_0 * _S12;
        d_0 = d_0 + u32(256);
    }
    return;
}

