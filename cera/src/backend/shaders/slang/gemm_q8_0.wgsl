struct GemmParams_std430_0
{
    @align(4) m_0 : u32,
    @align(4) k_0 : u32,
    @align(4) n_0 : u32,
    @align(4) x_stride_0 : u32,
    @align(4) y_stride_0 : u32,
    @align(4) _pad_0 : u32,
};

@binding(3) @group(0) var<storage, read> params_0 : array<GemmParams_std430_0>;

@binding(0) @group(0) var<storage, read> src0_0 : array<u32>;

@binding(1) @group(0) var<storage, read> src1_0 : array<f32>;

@binding(2) @group(0) var<storage, read_write> dst_0 : array<f32>;

fn load_f16_0( byte_off_0 : u32) -> f32
{
    var w_0 : u32 = src0_0[(byte_off_0 >> (u32(2)))];
    var h_0 : u32;
    if(((byte_off_0 & (u32(2)))) != u32(0))
    {
        h_0 = (w_0 >> (u32(16)));
    }
    else
    {
        h_0 = (w_0 & (u32(65535)));
    }
    return (unpack2x16float((h_0)).x);
}

fn load_byte_0( byte_off_1 : u32) -> u32
{
    return (((src0_0[(byte_off_1 >> (u32(2)))] >> ((((byte_off_1 & (u32(3)))) * u32(8))))) & (u32(255)));
}

fn load_i8_0( byte_off_2 : u32) -> i32
{
    return i32(((load_byte_0(byte_off_2)) ^ (u32(128)))) - i32(128);
}

@compute
@workgroup_size(128, 1, 1)
fn gemm_q8_0(@builtin(workgroup_id) gid_0 : vec3<u32>, @builtin(local_invocation_index) tiitg_0 : u32)
{
    var _S1 : GemmParams_std430_0 = params_0[i32(0)];
    var _S2 : GemmParams_std430_0 = params_0[i32(0)];
    var _S3 : GemmParams_std430_0 = params_0[i32(0)];
    var _S4 : GemmParams_std430_0 = params_0[i32(0)];
    var nb_0 : u32 = params_0[i32(0)].k_0 / u32(32);
    var row_bytes_0 : u32 = nb_0 * u32(34);
    var r0_0 : u32 = gid_0.y;
    var r1_0 : u32 = gid_0.x;
    var idx_0 : u32 = tiitg_0;
    for(;;)
    {
        if(idx_0 < u32(2048))
        {
        }
        else
        {
            break;
        }
        var row_0 : u32 = r0_0 * u32(64) + idx_0 % u32(64);
        var col_0 : u32 = r1_0 * u32(32) + idx_0 / u32(64);
        var _S5 : bool;
        if(row_0 >= (_S1.m_0))
        {
            _S5 = true;
        }
        else
        {
            _S5 = col_0 >= (_S2.n_0);
        }
        if(_S5)
        {
            idx_0 = idx_0 + u32(128);
            continue;
        }
        var _S6 : u32 = row_0 * row_bytes_0;
        var _S7 : u32 = col_0 * _S3.x_stride_0;
        var b_0 : u32 = u32(0);
        var acc_0 : f32 = 0.0f;
        for(;;)
        {
            if(b_0 < nb_0)
            {
            }
            else
            {
                break;
            }
            var blk_0 : u32 = _S6 + b_0 * u32(34);
            var _S8 : f32 = load_f16_0(blk_0);
            var e_0 : u32 = u32(0);
            for(;;)
            {
                if(e_0 < u32(32))
                {
                }
                else
                {
                    break;
                }
                var acc_1 : f32 = acc_0 + _S8 * f32(load_i8_0(blk_0 + u32(2) + e_0)) * src1_0[_S7 + b_0 * u32(32) + e_0];
                e_0 = e_0 + u32(1);
                acc_0 = acc_1;
            }
            b_0 = b_0 + u32(1);
        }
        dst_0[row_0 + col_0 * _S4.y_stride_0] = acc_0;
        idx_0 = idx_0 + u32(128);
    }
    return;
}

