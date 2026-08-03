@binding(1) @group(0) var<storage, read> par_buf_0 : array<vec2<u32>>;

@binding(0) @group(0) var<storage, read_write> x_buf_0 : array<f32>;

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
fn softmax(@builtin(local_invocation_id) lid_0 : vec3<u32>)
{
    var tid_2 : u32 = lid_0.x;
    var _S3 : u32 = par_buf_0[i32(0)].x;
    var local_max_0 : f32 = -3.4028234663852886e+38f;
    var i_0 : u32 = tid_2;
    for(;;)
    {
        if(i_0 < _S3)
        {
        }
        else
        {
            break;
        }
        var _S4 : f32 = max(local_max_0, x_buf_0[i_0]);
        var i_1 : u32 = i_0 + u32(256);
        local_max_0 = _S4;
        i_0 = i_1;
    }
    var _S5 : f32 = block_max_0(tid_2, local_max_0);
    workgroupBarrier();
    i_0 = tid_2;
    var partial_0 : f32 = 0.0f;
    for(;;)
    {
        if(i_0 < _S3)
        {
        }
        else
        {
            break;
        }
        var e_0 : f32 = exp(x_buf_0[i_0] - _S5);
        x_buf_0[i_0] = e_0;
        var partial_1 : f32 = partial_0 + e_0;
        i_0 = i_0 + u32(256);
        partial_0 = partial_1;
    }
    var _S6 : f32 = block_sum_0(tid_2, partial_0);
    var _S7 : f32 = 1.0f / _S6;
    i_0 = tid_2;
    for(;;)
    {
        if(i_0 < _S3)
        {
        }
        else
        {
            break;
        }
        x_buf_0[i_0] = x_buf_0[i_0] * _S7;
        i_0 = i_0 + u32(256);
    }
    return;
}

