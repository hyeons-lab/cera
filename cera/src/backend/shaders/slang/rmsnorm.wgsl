@binding(6) @group(0) var<storage, read> p_oop_0 : array<vec4<u32>>;

@binding(3) @group(0) var<storage, read> src_buf_0 : array<f32>;

@binding(4) @group(0) var<storage, read_write> dst_buf_0 : array<f32>;

@binding(5) @group(0) var<storage, read> w_oop_0 : array<f32>;

@binding(2) @group(0) var<storage, read> p_wgsl_0 : array<vec4<u32>>;

@binding(0) @group(0) var<storage, read_write> x_buf_0 : array<f32>;

@binding(1) @group(0) var<storage, read> w_wgsl_0 : array<f32>;

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
fn rmsnorm(@builtin(local_invocation_id) lid_0 : vec3<u32>)
{
    var tid_1 : u32 = lid_0.x;
    var n_0 : u32 = p_wgsl_0[i32(0)].x;
    var eps_0 : f32 = (bitcast<f32>((p_wgsl_0[i32(0)].y)));
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
        var partial_1 : f32 = partial_0 + x_buf_0[i_0] * x_buf_0[i_0];
        i_0 = i_0 + u32(256);
        partial_0 = partial_1;
    }
    var _S2 : f32 = block_sum_0(tid_1, partial_0);
    var _S3 : f32 = 1.0f / sqrt(_S2 / f32(n_0) + eps_0);
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
        x_buf_0[i_0] = x_buf_0[i_0] * _S3 * w_wgsl_0[i_0];
        i_0 = i_0 + u32(256);
    }
    return;
}

@compute
@workgroup_size(256, 1, 1)
fn rmsnorm_out(@builtin(local_invocation_id) lid_1 : vec3<u32>)
{
    var tid_2 : u32 = lid_1.x;
    var n_1 : u32 = p_oop_0[i32(0)].x;
    var eps_1 : f32 = (bitcast<f32>((p_oop_0[i32(0)].y)));
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
        var v_1 : f32 = src_buf_0[i_1];
        var partial_3 : f32 = partial_2 + v_1 * v_1;
        i_1 = i_1 + u32(256);
        partial_2 = partial_3;
    }
    var _S4 : f32 = block_sum_0(tid_2, partial_2);
    var _S5 : f32 = 1.0f / sqrt(_S4 / f32(n_1) + eps_1);
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
        dst_buf_0[i_1] = src_buf_0[i_1] * _S5 * w_oop_0[i_1];
        i_1 = i_1 + u32(256);
    }
    return;
}

