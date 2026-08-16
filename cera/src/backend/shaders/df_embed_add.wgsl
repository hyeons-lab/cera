// Fused depthformer codebook embedding lookup + add to hidden state.
// Reads sampled token `code = sampled_codes[code_idx]`, looks up row `code` in
// `codebook_emb` [n_vocab x emb_dim], and adds it in-place to `hidden` [emb_dim].

@binding(0) @group(0) var<storage, read> codebook_emb : array<f32>;
@binding(1) @group(0) var<storage, read> sampled_codes : array<u32>;
@binding(2) @group(0) var<storage, read_write> hidden : array<f32>;
@binding(3) @group(0) var<storage, read> params : array<vec2<u32>>; // [emb_dim, code_idx]

@compute
@workgroup_size(256, 1, 1)
fn df_embed_add(@builtin(global_invocation_id) gid : vec3<u32>) {
    let emb_dim = params[0].x;
    let code_idx = params[0].y;
    let code = sampled_codes[code_idx];
    let i = gid.x;
    if (i < emb_dim) {
        let offset = code * emb_dim + i;
        hidden[i] = hidden[i] + codebook_emb[offset];
    }
}
