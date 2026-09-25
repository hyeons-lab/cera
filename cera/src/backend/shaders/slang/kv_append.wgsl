@binding(2) @group(0) var<storage, read> par_buf_0 : array<vec4<u32>>;

@binding(1) @group(0) var<storage, read_write> dst_buf_0 : array<u32>;

@binding(0) @group(0) var<storage, read> src_buf_0 : array<f32>;

fn to_f16_bits_0( v_0 : f32) -> u32
{
    var b_0 : u32 = (bitcast<u32>((v_0)));
    var s_0 : u32 = (((b_0 >> (u32(16)))) & (u32(32768)));
    var e_0 : i32 = i32((((b_0 >> (u32(23)))) & (u32(255)))) - i32(112);
    var m_0 : u32 = (b_0 & (u32(8388607)));
    var h_0 : u32;
    if(e_0 >= i32(31))
    {
        h_0 = u32(31743);
    }
    else
    {
        if(e_0 <= i32(0))
        {
            h_0 = u32(0);
        }
        else
        {
            h_0 = (((u32(e_0) << (u32(10)))) | ((((m_0 + u32(4096)) >> (u32(13))))));
        }
    }
    return (s_0 | (h_0));
}

@compute
@workgroup_size(256, 1, 1)
fn kv_append(@builtin(global_invocation_id) gid_0 : vec3<u32>)
{
    var w_0 : u32 = gid_0.x;
    var off_0 : u32 = par_buf_0[i32(0)].x;
    if(w_0 >= (((par_buf_0[i32(0)].y) >> (u32(1)))))
    {
        return;
    }
    var i0_0 : u32 = (w_0 << (u32(1)));
    dst_buf_0[off_0 + w_0] = ((to_f16_bits_0(src_buf_0[i0_0])) | ((((to_f16_bits_0(src_buf_0[i0_0 + u32(1)])) << (u32(16))))));
    return;
}

