@binding(2) @group(0) var<storage, read> par_buf_0 : array<vec4<u32>>;

@binding(1) @group(0) var<storage, read_write> dst_buf_0 : array<f32>;

@binding(0) @group(0) var<storage, read> mel_buf_0 : array<f32>;

var<workgroup> scratch_0 : array<f32, i32(256)>;

fn block_sum_0( tid_0 : u32,  v_0 : f32) -> f32
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
            scratch_0[tid_0] = scratch_0[tid_0] + scratch_0[tid_0 + s_0];
        }
        workgroupBarrier();
        s_0 = (s_0 >> (u32(1)));
    }
    var _S1 : f32 = scratch_0[i32(0)];
    return _S1;
}

@compute
@workgroup_size(256, 1, 1)
fn mel_norm(@builtin(local_invocation_id) lid_0 : vec3<u32>, @builtin(workgroup_id) wid_0 : vec3<u32>)
{
    var tid_1 : u32 = lid_0.x;
    var mi_0 : u32 = wid_0.x;
    var _S2 : u32 = par_buf_0[i32(0)].x;
    var n_frames_0 : u32 = par_buf_0[i32(0)].y;
    var eff_0 : u32 = par_buf_0[i32(0)].z;
    var eps_0 : f32 = (bitcast<f32>((par_buf_0[i32(0)].w)));
    var _S3 : u32 = mi_0 * n_frames_0;
    var t_0 : u32;
    if(eff_0 <= u32(1))
    {
        t_0 = tid_1;
        for(;;)
        {
            if(t_0 < n_frames_0)
            {
            }
            else
            {
                break;
            }
            dst_buf_0[t_0 * _S2 + mi_0] = 0.0f;
            t_0 = t_0 + u32(256);
        }
        return;
    }
    t_0 = tid_1;
    var partial_0 : f32 = 0.0f;
    for(;;)
    {
        if(t_0 < eff_0)
        {
        }
        else
        {
            break;
        }
        var partial_1 : f32 = partial_0 + mel_buf_0[_S3 + t_0];
        t_0 = t_0 + u32(256);
        partial_0 = partial_1;
    }
    var _S4 : f32 = block_sum_0(tid_1, partial_0);
    var _S5 : f32 = _S4 / f32(eff_0);
    workgroupBarrier();
    t_0 = tid_1;
    partial_0 = 0.0f;
    for(;;)
    {
        if(t_0 < eff_0)
        {
        }
        else
        {
            break;
        }
        var d_0 : f32 = mel_buf_0[_S3 + t_0] - _S5;
        var partial_2 : f32 = partial_0 + d_0 * d_0;
        t_0 = t_0 + u32(256);
        partial_0 = partial_2;
    }
    var _S6 : f32 = block_sum_0(tid_1, partial_0);
    var _S7 : f32 = 1.0f / sqrt(_S6 / f32(eff_0 - u32(1)) + eps_0);
    t_0 = tid_1;
    for(;;)
    {
        if(t_0 < n_frames_0)
        {
        }
        else
        {
            break;
        }
        var v_1 : f32;
        if(t_0 < eff_0)
        {
            v_1 = (mel_buf_0[_S3 + t_0] - _S5) * _S7;
        }
        else
        {
            v_1 = 0.0f;
        }
        dst_buf_0[t_0 * _S2 + mi_0] = v_1;
        t_0 = t_0 + u32(256);
    }
    return;
}

