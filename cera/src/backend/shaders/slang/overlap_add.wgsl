@binding(3) @group(0) var<storage, read> par_buf_0 : array<vec4<u32>>;

@binding(1) @group(0) var<storage, read> hann_buf_0 : array<f32>;

@binding(0) @group(0) var<storage, read> td_buf_0 : array<f32>;

@binding(2) @group(0) var<storage, read_write> out_buf_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn overlap_add(@builtin(global_invocation_id) gid_0 : vec3<u32>)
{
    var n_frames_0 : u32 = par_buf_0[i32(0)].x;
    var n_fft_0 : u32 = par_buf_0[i32(0)].y;
    var hop_0 : u32 = par_buf_0[i32(0)].z;
    var g_0 : u32 = gid_0.x;
    if(g_0 >= (n_frames_0 * hop_0))
    {
        return;
    }
    var i_hi_0 : u32 = g_0 / hop_0;
    var i_hi_1 : u32;
    if(i_hi_0 >= n_frames_0)
    {
        i_hi_1 = n_frames_0 - u32(1);
    }
    else
    {
        i_hi_1 = i_hi_0;
    }
    var i_lo_0 : u32;
    if(g_0 >= n_fft_0)
    {
        var _S1 : u32 = (g_0 - n_fft_0) / hop_0;
        i_lo_0 = _S1 + u32(1);
    }
    else
    {
        i_lo_0 = u32(0);
    }
    var i_0 : u32 = i_lo_0;
    var numer_0 : f32 = 0.0f;
    var denom_0 : f32 = 0.0f;
    for(;;)
    {
        if(i_0 <= i_hi_1)
        {
        }
        else
        {
            break;
        }
        var local_0 : u32 = g_0 - i_0 * hop_0;
        var w_0 : f32 = hann_buf_0[local_0];
        var numer_1 : f32 = numer_0 + td_buf_0[i_0 * n_fft_0 + local_0] * w_0;
        var denom_1 : f32 = denom_0 + w_0 * w_0;
        i_0 = i_0 + u32(1);
        numer_0 = numer_1;
        denom_0 = denom_1;
    }
    if(denom_0 > 9.99999993922529029e-09f)
    {
        numer_0 = numer_0 / denom_0;
    }
    else
    {
    }
    out_buf_0[g_0] = numer_0;
    return;
}

