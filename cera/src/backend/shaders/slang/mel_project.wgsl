@binding(3) @group(0) var<storage, read> par_buf_0 : array<vec4<u32>>;

@binding(0) @group(0) var<storage, read> power_buf_0 : array<f32>;

@binding(1) @group(0) var<storage, read> filter_buf_0 : array<f32>;

@binding(2) @group(0) var<storage, read_write> mel_buf_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn mel_project(@builtin(global_invocation_id) gid_0 : vec3<u32>)
{
    var n_frames_0 : u32 = par_buf_0[i32(0)].y;
    var n_bins_0 : u32 = par_buf_0[i32(0)].z;
    var eps_0 : f32 = (bitcast<f32>((par_buf_0[i32(0)].w)));
    var idx_0 : u32 = gid_0.x;
    if(idx_0 >= (par_buf_0[i32(0)].x * n_frames_0))
    {
        return;
    }
    var mi_0 : u32 = idx_0 / n_frames_0;
    var _S1 : u32 = (idx_0 - mi_0 * n_frames_0) * n_bins_0;
    var _S2 : u32 = mi_0 * n_bins_0;
    var k_0 : u32 = u32(0);
    var sum_0 : f32 = 0.0f;
    for(;;)
    {
        if(k_0 < n_bins_0)
        {
        }
        else
        {
            break;
        }
        var sum_1 : f32 = sum_0 + power_buf_0[_S1 + k_0] * filter_buf_0[_S2 + k_0];
        k_0 = k_0 + u32(1);
        sum_0 = sum_1;
    }
    mel_buf_0[idx_0] = log(sum_0 + eps_0);
    return;
}

