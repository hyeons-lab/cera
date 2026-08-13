@binding(4) @group(0) var<storage, read> params_0 : array<vec4<u32>>;

@binding(0) @group(0) var<storage, read> logits_0 : array<f32>;

@binding(1) @group(0) var<storage, read> bias_0 : array<f32>;

@binding(2) @group(0) var<storage, read_write> sel_expert_0 : array<u32>;

@binding(3) @group(0) var<storage, read_write> sel_weight_0 : array<f32>;

var<workgroup> sh_prob_0 : array<f32, i32(256)>;

var<workgroup> sh_score_0 : array<f32, i32(256)>;

@compute
@workgroup_size(32, 1, 1)
fn moe_route(@builtin(local_invocation_id) lid_0 : vec3<u32>, @builtin(workgroup_id) grp_0 : vec3<u32>)
{
    var n_expert_0 : u32 = params_0[i32(0)].x;
    var _S1 : u32 = min(min(params_0[i32(0)].y, n_expert_0), u32(16));
    var tok_0 : u32 = grp_0.x;
    if(tok_0 >= (params_0[i32(0)].z))
    {
        return;
    }
    var _S2 : u32 = lid_0.x;
    var e_0 : u32 = _S2;
    for(;;)
    {
        if(e_0 < n_expert_0)
        {
        }
        else
        {
            break;
        }
        var p_0 : f32 = 1.0f / (1.0f + exp(- logits_0[tok_0 * n_expert_0 + e_0]));
        sh_prob_0[e_0] = p_0;
        sh_score_0[e_0] = p_0 + bias_0[e_0];
        e_0 = e_0 + u32(32);
    }
    workgroupBarrier();
    if(_S2 != u32(0))
    {
        return;
    }
    var chosen_0 : array<u32, i32(16)>;
    var unnorm_w_0 : array<f32, i32(16)>;
    var i_0 : u32 = u32(0);
    for(;;)
    {
        if(i_0 < u32(16))
        {
        }
        else
        {
            break;
        }
        chosen_0[i_0] = u32(4294967295);
        unnorm_w_0[i_0] = 0.0f;
        i_0 = i_0 + u32(1);
    }
    var s_0 : u32 = u32(0);
    var sum_0 : f32 = 0.0f;
    for(;;)
    {
        if(s_0 < _S1)
        {
        }
        else
        {
            break;
        }
        var have_0 : bool = false;
        var best_score_0 : f32 = 0.0f;
        var best_0 : u32 = u32(0);
        e_0 = u32(0);
        for(;;)
        {
            if(e_0 < n_expert_0)
            {
            }
            else
            {
                break;
            }
            var used_0 : bool = false;
            var t_0 : u32 = u32(0);
            for(;;)
            {
                if(t_0 < s_0)
                {
                }
                else
                {
                    break;
                }
                if((chosen_0[t_0]) == e_0)
                {
                    used_0 = true;
                }
                t_0 = t_0 + u32(1);
            }
            if(used_0)
            {
                e_0 = e_0 + u32(1);
                continue;
            }
            var score_0 : f32 = sh_score_0[e_0];
            var _S3 : bool;
            if(!have_0)
            {
                _S3 = true;
            }
            else
            {
                _S3 = score_0 > best_score_0;
            }
            var best_1 : u32;
            var best_score_1 : f32;
            var have_1 : bool;
            if(_S3)
            {
                have_1 = true;
                best_score_1 = score_0;
                best_1 = e_0;
            }
            else
            {
                have_1 = have_0;
                best_score_1 = best_score_0;
                best_1 = best_0;
            }
            have_0 = have_1;
            best_score_0 = best_score_1;
            best_0 = best_1;
            e_0 = e_0 + u32(1);
        }
        chosen_0[s_0] = best_0;
        unnorm_w_0[s_0] = sh_prob_0[best_0];
        sel_expert_0[tok_0 * _S1 + s_0] = best_0;
        var sum_1 : f32 = sum_0 + sh_prob_0[best_0];
        s_0 = s_0 + u32(1);
        sum_0 = sum_1;
    }
    var _S4 : f32 = 1.0f / max(sum_0, 0.00006103515625f);
    s_0 = u32(0);
    for(;;)
    {
        if(s_0 < _S1)
        {
        }
        else
        {
            break;
        }
        sel_weight_0[tok_0 * _S1 + s_0] = unnorm_w_0[s_0] * _S4;
        s_0 = s_0 + u32(1);
    }
    return;
}

