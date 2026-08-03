@binding(4) @group(0) var<storage, read> par_buf_0 : array<vec4<u32>>;

@binding(0) @group(0) var<storage, read> src_buf_0 : array<f32>;

@binding(1) @group(0) var<storage, read_write> dst_buf_0 : array<f32>;

@binding(2) @group(0) var<storage, read> weight_buf_0 : array<f32>;

@binding(3) @group(0) var<storage, read> bias_buf_0 : array<f32>;

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
fn layernorm_batch(@builtin(local_invocation_id) lid_0 : vec3<u32>, @builtin(workgroup_id) wid_0 : vec3<u32>)
{
    var tid_1 : u32 = lid_0.x;
    var row_0 : u32 = wid_0.x;
    var n_0 : u32 = par_buf_0[i32(0)].x;
    var eps_0 : f32 = (bitcast<f32>((par_buf_0[i32(0)].y)));
    var _S2 : u32 = row_0 * par_buf_0[i32(0)].z;
    var _S3 : u32 = row_0 * par_buf_0[i32(0)].w;
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
        var partial_1 : f32 = partial_0 + src_buf_0[_S2 + i_0];
        i_0 = i_0 + u32(256);
        partial_0 = partial_1;
    }
    var _S4 : f32 = block_sum_0(tid_1, partial_0);
    var _S5 : f32 = f32(n_0);
    var _S6 : f32 = _S4 / _S5;
    workgroupBarrier();
    i_0 = tid_1;
    partial_0 = 0.0f;
    for(;;)
    {
        if(i_0 < n_0)
        {
        }
        else
        {
            break;
        }
        var d_0 : f32 = src_buf_0[_S2 + i_0] - _S6;
        var partial_2 : f32 = partial_0 + d_0 * d_0;
        i_0 = i_0 + u32(256);
        partial_0 = partial_2;
    }
    var _S7 : f32 = block_sum_0(tid_1, partial_0);
    var _S8 : f32 = 1.0f / sqrt(_S7 / _S5 + eps_0);
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
        dst_buf_0[_S3 + i_0] = (src_buf_0[_S2 + i_0] - _S6) * _S8 * weight_buf_0[i_0] + bias_buf_0[i_0];
        i_0 = i_0 + u32(256);
    }
    return;
}

