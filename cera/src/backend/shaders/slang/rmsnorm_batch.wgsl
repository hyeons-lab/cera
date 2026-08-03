@binding(3) @group(0) var<storage, read> par_buf_0 : array<u32>;

@binding(0) @group(0) var<storage, read_write> src_buf_0 : array<f32>;

@binding(1) @group(0) var<storage, read_write> dst_buf_0 : array<f32>;

@binding(2) @group(0) var<storage, read> w_buf_0 : array<f32>;

@binding(4) @group(0) var<storage, read> res_buf_0 : array<f32>;

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
fn rmsnorm_batch(@builtin(local_invocation_id) lid_0 : vec3<u32>, @builtin(workgroup_id) wid_0 : vec3<u32>)
{
    var tid_1 : u32 = lid_0.x;
    var n_0 : u32 = par_buf_0[i32(0)];
    var eps_0 : f32 = (bitcast<f32>((par_buf_0[i32(1)])));
    var _S2 : u32 = wid_0.x;
    var _S3 : u32 = _S2 * par_buf_0[i32(2)];
    var _S4 : u32 = _S2 * par_buf_0[i32(3)];
    var i_0 : u32 = tid_1;
    var partial_0 : f32 = 0.0f;
    for(;;)
    {
        if(i_0 < n_0)
        {
        }
        else
        {
            break;
        }
        var partial_1 : f32 = partial_0 + src_buf_0[_S3 + i_0] * src_buf_0[_S3 + i_0];
        i_0 = i_0 + u32(256);
        partial_0 = partial_1;
    }
    var _S5 : f32 = block_sum_0(tid_1, partial_0);
    var _S6 : f32 = 1.0f / sqrt(_S5 / f32(n_0) + eps_0);
    i_0 = tid_1;
    for(;;)
    {
        if(i_0 < n_0)
        {
        }
        else
        {
            break;
        }
        dst_buf_0[_S4 + i_0] = src_buf_0[_S3 + i_0] * _S6 * w_buf_0[i_0];
        i_0 = i_0 + u32(256);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn add_rmsnorm_batch(@builtin(local_invocation_id) lid_1 : vec3<u32>, @builtin(workgroup_id) wid_1 : vec3<u32>)
{
    var tid_2 : u32 = lid_1.x;
    var n_1 : u32 = par_buf_0[i32(0)];
    var eps_1 : f32 = (bitcast<f32>((par_buf_0[i32(1)])));
    var _S7 : u32 = wid_1.x;
    var _S8 : u32 = _S7 * par_buf_0[i32(2)];
    var _S9 : u32 = _S7 * par_buf_0[i32(3)];
    var _S10 : f32 = (bitcast<f32>((par_buf_0[i32(4)])));
    var i_1 : u32 = tid_2;
    var partial_2 : f32 = 0.0f;
    for(;;)
    {
        if(i_1 < n_1)
        {
        }
        else
        {
            break;
        }
        var _S11 : u32 = _S8 + i_1;
        var v_1 : f32 = src_buf_0[_S11] + _S10 * res_buf_0[_S11];
        src_buf_0[_S11] = v_1;
        var partial_3 : f32 = partial_2 + v_1 * v_1;
        i_1 = i_1 + u32(256);
        partial_2 = partial_3;
    }
    var _S12 : f32 = block_sum_0(tid_2, partial_2);
    var _S13 : f32 = 1.0f / sqrt(_S12 / f32(n_1) + eps_1);
    i_1 = tid_2;
    for(;;)
    {
        if(i_1 < n_1)
        {
        }
        else
        {
            break;
        }
        dst_buf_0[_S9 + i_1] = src_buf_0[_S8 + i_1] * _S13 * w_buf_0[i_1];
        i_1 = i_1 + u32(256);
    }
    return;
}

