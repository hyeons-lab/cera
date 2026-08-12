@binding(3) @group(0) var<storage, read> par_buf_0 : array<vec4<u32>>;

@binding(0) @group(0) var<storage, read> pcm_buf_0 : array<f32>;

@binding(2) @group(0) var<storage, read_write> frames_buf_0 : array<f32>;

@binding(1) @group(0) var<storage, read> hann_buf_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn stft_frame(@builtin(global_invocation_id) gid_0 : vec3<u32>)
{
    var n_fft_0 : u32 = par_buf_0[i32(0)].y;
    var hop_0 : u32 = par_buf_0[i32(0)].z;
    var center_pad_0 : u32 = par_buf_0[i32(0)].w;
    var n_samples_0 : u32 = par_buf_0[i32(1)].x;
    var preemph_0 : f32 = (bitcast<f32>((par_buf_0[i32(1)].y)));
    var idx_0 : u32 = gid_0.x;
    if(idx_0 >= (par_buf_0[i32(0)].x * n_fft_0))
    {
        return;
    }
    var t_0 : u32 = idx_0 / n_fft_0;
    var n_0 : u32 = idx_0 - t_0 * n_fft_0;
    var g_0 : i32 = i32(t_0 * hop_0 + n_0) - i32(center_pad_0);
    var _S1 : bool;
    if(g_0 >= i32(0))
    {
        _S1 = g_0 < i32(n_samples_0);
    }
    else
    {
        _S1 = false;
    }
    var s_0 : f32;
    if(_S1)
    {
        var s_1 : f32 = pcm_buf_0[g_0];
        if(g_0 > i32(0))
        {
            s_0 = s_1 - preemph_0 * pcm_buf_0[g_0 - i32(1)];
        }
        else
        {
            s_0 = s_1;
        }
    }
    else
    {
        s_0 = 0.0f;
    }
    frames_buf_0[idx_0] = hann_buf_0[n_0] * s_0;
    return;
}

