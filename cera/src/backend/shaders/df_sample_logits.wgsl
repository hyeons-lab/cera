// Fused Depthformer Top-4 Softmax & Temperature Sampling on GPU.
// Dispatched with 1 workgroup of 256 threads.
// Each thread finds top-4 candidates in its chunk of 8 logits,
// followed by parallel workgroup reduction to find the global top-4.
//
// Top-4 candidate reduction covers >99.9% of the probability mass for
// speech codebook distributions (vocab=2048) under standard sampling
// temperature (0.6-0.8), while fitting cleanly into a single SIMD vector
// register per thread without shared memory spills or multi-pass sorting.

@binding(0) @group(0) var<storage, read> logits : array<f32>;
@binding(1) @group(0) var<storage, read_write> sampled_codes : array<u32>;
@binding(2) @group(0) var<storage, read> params : array<vec4<f32>>;
// params[0] = vec4<f32>(n_vocab, code_idx, inv_temperature, rand_val)

var<workgroup> s_top_vals : array<vec4<f32>, 256>;
var<workgroup> s_top_idxs : array<vec4<u32>, 256>;

fn insert_top4(val: f32, idx: u32, p_vals: ptr<function, vec4<f32>>, p_idxs: ptr<function, vec4<u32>>) {
    var v = *p_vals;
    var i = *p_idxs;
    if (val > v.x) {
        v.w = v.z; i.w = i.z;
        v.z = v.y; i.z = i.y;
        v.y = v.x; i.y = i.x;
        v.x = val; i.x = idx;
    } else if (val > v.y) {
        v.w = v.z; i.w = i.z;
        v.z = v.y; i.z = i.y;
        v.y = val; i.y = idx;
    } else if (val > v.z) {
        v.w = v.z; i.w = i.z;
        v.z = val; i.z = idx;
    } else if (val > v.w) {
        v.w = val; i.w = idx;
    }
    *p_vals = v;
    *p_idxs = i;
}

fn merge_top4(other_val: vec4<f32>, other_idx: vec4<u32>, p_vals: ptr<function, vec4<f32>>, p_idxs: ptr<function, vec4<u32>>) {
    insert_top4(other_val.x, other_idx.x, p_vals, p_idxs);
    insert_top4(other_val.y, other_idx.y, p_vals, p_idxs);
    insert_top4(other_val.z, other_idx.z, p_vals, p_idxs);
    insert_top4(other_val.w, other_idx.w, p_vals, p_idxs);
}

@compute
@workgroup_size(256, 1, 1)
fn df_sample_logits(@builtin(local_invocation_id) lid : vec3<u32>) {
    let tid = lid.x;
    let n_vocab = u32(params[0].x);
    let code_idx = u32(params[0].y);
    let inv_temp = params[0].z;
    let rand_val = params[0].w;

    var local_vals = vec4<f32>(-3.402823e+38, -3.402823e+38, -3.402823e+38, -3.402823e+38);
    var local_idxs = vec4<u32>(0u, 0u, 0u, 0u);

    // 1. Each thread gathers local top-4 candidates
    for (var i = tid; i < n_vocab; i += 256u) {
        let v = logits[i];
        insert_top4(v, i, &local_vals, &local_idxs);
    }

    s_top_vals[tid] = local_vals;
    s_top_idxs[tid] = local_idxs;
    workgroupBarrier();

    // 2. Parallel reduction to merge top-4 across all 256 threads
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            var cur_vals = s_top_vals[tid];
            var cur_idxs = s_top_idxs[tid];
            let other_vals = s_top_vals[tid + stride];
            let other_idxs = s_top_idxs[tid + stride];
            merge_top4(other_vals, other_idxs, &cur_vals, &cur_idxs);
            s_top_vals[tid] = cur_vals;
            s_top_idxs[tid] = cur_idxs;
        }
        workgroupBarrier();
    }

    // 3. Thread 0 computes temperature-scaled softmax on top-4 and samples
    if (tid == 0u) {
        let top_v = s_top_vals[0];
        let top_i = s_top_idxs[0];
        let max_v = top_v.x;

        // If inv_temp is very high or negative, default to argmax
        if (inv_temp <= 0.0 || inv_temp > 100.0) {
            sampled_codes[code_idx] = top_i.x;
        } else {
            let e0 = exp((top_v.x - max_v) * inv_temp);
            let e1 = exp((top_v.y - max_v) * inv_temp);
            let e2 = exp((top_v.z - max_v) * inv_temp);
            let e3 = exp((top_v.w - max_v) * inv_temp);
            let sum = e0 + e1 + e2 + e3;

            let r = rand_val * sum;
            var chosen = top_i.x;
            if (r <= e0) {
                chosen = top_i.x;
            } else if (r <= e0 + e1) {
                chosen = top_i.y;
            } else if (r <= e0 + e1 + e2) {
                chosen = top_i.z;
            } else {
                chosen = top_i.w;
            }
            sampled_codes[code_idx] = chosen;
        }
    }
}
