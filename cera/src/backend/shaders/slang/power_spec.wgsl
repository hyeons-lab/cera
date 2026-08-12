@binding(3) @group(0) var<storage, read> par_buf_0 : array<vec4<u32>>;

@binding(0) @group(0) var<storage, read> frames_buf_0 : array<f32>;

@binding(1) @group(0) var<storage, read> twiddle_buf_0 : array<f32>;

@binding(2) @group(0) var<storage, read_write> power_buf_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn power_spec(@builtin(global_invocation_id) gid_0 : vec3<u32>)
{
    var n_fft_0 : u32 = par_buf_0[i32(0)].y;
    var n_bins_0 : u32 = par_buf_0[i32(0)].z;
    var idx_0 : u32 = gid_0.x;
    if(idx_0 >= (par_buf_0[i32(0)].x * n_bins_0))
    {
        return;
    }
    var t_0 : u32 = idx_0 / n_bins_0;
    var _S1 : u32 = idx_0 - t_0 * n_bins_0;
    var _S2 : u32 = t_0 * n_fft_0;
    var n_0 : u32 = u32(0);
    var m_0 : u32 = u32(0);
    var re_0 : f32 = 0.0f;
    var im_0 : f32 = 0.0f;
    for(;;)
    {
        if(n_0 < n_fft_0)
        {
        }
        else
        {
            break;
        }
        var x_0 : f32 = frames_buf_0[_S2 + n_0];
        var _S3 : u32 = u32(2) * m_0;
        var re_1 : f32 = re_0 + x_0 * twiddle_buf_0[_S3];
        var im_1 : f32 = im_0 + x_0 * twiddle_buf_0[_S3 + u32(1)];
        var m_1 : u32 = m_0 + _S1;
        if(m_1 >= n_fft_0)
        {
            m_0 = m_1 - n_fft_0;
        }
        else
        {
            m_0 = m_1;
        }
        n_0 = n_0 + u32(1);
        re_0 = re_1;
        im_0 = im_1;
    }
    power_buf_0[idx_0] = re_0 * re_0 + im_0 * im_0;
    return;
}

