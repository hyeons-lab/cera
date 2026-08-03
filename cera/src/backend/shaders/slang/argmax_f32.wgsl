@binding(2) @group(0) var<storage, read> par_buf_0 : array<vec2<u32>>;

@binding(0) @group(0) var<storage, read> x_buf_0 : array<f32>;

@binding(1) @group(0) var<storage, read_write> out_buf_0 : array<u32>;

var<workgroup> scratch_v_0 : array<f32, i32(256)>;

var<workgroup> scratch_i_0 : array<u32, i32(256)>;

fn block_argmax_0( tid_0 : u32,  val_0 : f32,  idx_0 : u32) -> u32
{
    scratch_v_0[tid_0] = val_0;
    scratch_i_0[tid_0] = idx_0;
    workgroupBarrier();
    var stride_0 : u32 = u32(128);
    for(;;)
    {
        if(stride_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(tid_0 < stride_0)
        {
            var _S1 : u32 = tid_0 + stride_0;
            var ov_0 : f32 = scratch_v_0[_S1];
            var oi_0 : u32 = scratch_i_0[_S1];
            if((scratch_v_0[_S1]) > (scratch_v_0[tid_0]))
            {
                scratch_v_0[tid_0] = ov_0;
                scratch_i_0[tid_0] = oi_0;
            }
            else
            {
                var _S2 : bool;
                if(ov_0 == (scratch_v_0[tid_0]))
                {
                    _S2 = oi_0 < (scratch_i_0[tid_0]);
                }
                else
                {
                    _S2 = false;
                }
                if(_S2)
                {
                    scratch_i_0[tid_0] = oi_0;
                }
            }
        }
        workgroupBarrier();
        stride_0 = (stride_0 >> (u32(1)));
    }
    var _S3 : u32 = scratch_i_0[i32(0)];
    return _S3;
}

@compute
@workgroup_size(256, 1, 1)
fn argmax_f32(@builtin(local_invocation_id) lid_0 : vec3<u32>)
{
    var tid_1 : u32 = lid_0.x;
    var _S4 : u32 = par_buf_0[i32(0)].x;
    var local_max_0 : f32 = -3.4028234663852886e+38f;
    var local_idx_0 : u32 = u32(0);
    var i_0 : u32 = tid_1;
    for(;;)
    {
        if(i_0 < _S4)
        {
        }
        else
        {
            break;
        }
        var v_0 : f32 = x_buf_0[i_0];
        if(v_0 > local_max_0)
        {
            local_max_0 = v_0;
            local_idx_0 = i_0;
        }
        i_0 = i_0 + u32(256);
    }
    var winner_0 : u32 = block_argmax_0(tid_1, local_max_0, local_idx_0);
    if(tid_1 == u32(0))
    {
        out_buf_0[i32(0)] = winner_0;
    }
    return;
}

