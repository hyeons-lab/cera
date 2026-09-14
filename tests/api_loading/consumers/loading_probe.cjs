const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const api = require(path.resolve(process.argv[2]));
const root = process.argv[3];
const read = name => new Uint8Array(fs.readFileSync(path.join(root, name)));
const config = (backend = 'cpu', parts = false) => {
    const value = new api.LoadConfig(24, backend);
    value.draft_model = parts ? 'missing-probe-draft.gguf' : undefined;
    value.gpu_depthformer = parts;
    return value;
};
function consumed(loader) {
    assert.throws(() => loader.build(), error => error.code === 'Consumed');
    assert.throws(() => loader.buildGenerative(), error => error.code === 'Consumed');
}
function checkInfo(model, parts = false) {
    const info = api.info_for_probe(model);
    assert.equal(info.requested_context, 24);
    assert.equal(info.capacity, 24);
    assert.equal(info.backend, 'Cpu');
    assert.equal(info.gpu_depthformer, parts);
    assert.equal(info.draft_model, parts ? 'missing-probe-draft.gguf' : undefined);
    if (parts) {
        assert.equal(info.chat_template, 'probe-template');
        for (const [key, value] of Object.entries({temperature: 0.37, top_p: 0.71, min_p: 0.13, repetition_penalty: 1.23})) {
            assert.ok(Math.abs(info[key] - value) < 0.00001);
        }
        assert.equal(info.top_k, 7);
    }
    info.free();
}
function newSession(model) {
    const cfg = new api.SessionConfig();
    cfg.seed = 42n;
    try { return model.createSession(cfg); } finally { cfg.free(); }
}
function runSession(session) {
    session.appendTokens(new Uint32Array([0, 1]));
    assert.equal(session.position, 2);
    const opts = new api.GenerateOpts();
    Object.assign(opts, {maxTokens: 3, temperature: 0, ignoreEos: true});
    const tokens = [];
    const summary = session.generate(opts, batch => tokens.push(...batch));
    assert.equal(summary.tokensGenerated, 3);
    assert.equal(tokens.length, 3);
    assert.ok(tokens.every(token => token < 2));
    const position = session.position;
    assert.equal(position, 5);
    summary.free();
    opts.free();
    session.free();
    return {tokens, position};
}

const cases = [];
// Exercise the native checked conversion at actual 32-bit pointer width.
// This diagnostic uses BigInt; web LoadConfig still uses its u32 Number field.
for (const [request, expected] of [[0n, 0xffffffffn], [24n, 24n], [0xffffffffn, 0xffffffffn]]) {
    assert.equal(api.native_context_size_for_probe(request), expected);
}
for (const request of [0x100000000n, 0xffffffffffffffffn]) {
    assert.throws(() => api.native_context_size_for_probe(request), error =>
        error.code === 'InvalidConfig' && error.field === 'context_size'
        && error.value === request.toString() && error.reason === 'out_of_range' && !!error.message);
}
cases.push('native-context-32');
const input = read('model.gguf');
const source = api.ModelSource.bytes(input);
input.fill(0);
// Source and config transfer ownership into the loader; do not free them again.
const loader = new api.ModelLoader(source, config());
const handle = loader.build();
consumed(loader);
assert.equal(handle.kind(), 'Generative');
const first = handle.asGenerative();
const second = handle.asGenerative();
assert.ok(first && second);
checkInfo(second);
loader.free();
handle.free();
first.free();
const session = newSession(second);
second.free();
const generation = runSession(session);
cases.push('bytes-lifetime');

const parts = new api.ModelParts(read('model.gguf'));
parts.multimodal_projector = new Uint8Array([1]);
parts.audio_decoder = new Uint8Array([2]);
parts.audio_tokenizer = new Uint8Array([3]);
parts.draft_model = new Uint8Array([4]);
parts.inference_type = 'llama.cpp/text-to-text';
parts.chat_template = 'probe-template';
const defaults = new api.SamplingDefaults();
Object.assign(defaults, {temperature: 0.37, top_p: 0.71, top_k: 7, min_p: 0.13, repetition_penalty: 1.23});
parts.generation_defaults = api.GenerationDefaults.text(defaults);
const partsLoader = new api.ModelLoader(api.ModelSource.parts(parts), config('cpu', true));
const partsModel = partsLoader.buildGenerative();
consumed(partsLoader);
checkInfo(partsModel, true);
runSession(newSession(partsModel));
partsModel.free();
partsLoader.free();
cases.push('parts-defaults');

const emptySampling = () => new api.SamplingDefaults();
const fullSampling = () => Object.assign(emptySampling(), {
    temperature: 0.375, top_p: 0.75, top_k: 7, min_p: 0.125, repetition_penalty: 1.25,
});
const profiles = [
    ['absent', () => undefined, {kind: 'Text', parameters: {sampling_parameters: {}}}],
    ['text-empty', () => api.GenerationDefaults.text(emptySampling()),
        {kind: 'Text', parameters: {sampling_parameters: {}}}],
    ['audio', () => api.GenerationDefaults.audio(fullSampling(), 3, 0.625, 11),
        {kind: 'Audio', parameters: {number_of_decoding_threads: 3, audio_temperature: 0.625,
            audio_top_k: 11, temperature: 0.375, top_p: 0.75, top_k: 7,
            min_p: 0.125, repetition_penalty: 1.25}}],
    ['audio', () => api.GenerationDefaults.audio(emptySampling(), 0, 0, 0),
        {kind: 'Audio', parameters: {number_of_decoding_threads: 0, audio_temperature: 0, audio_top_k: 0}}],
    ['audio', () => api.GenerationDefaults.audio(emptySampling(), 0xffffffff, 1, 0xffffffff),
        {kind: 'Audio', parameters: {number_of_decoding_threads: 0xffffffff, audio_temperature: 1,
            audio_top_k: 0xffffffff}}],
    ['audio-empty', () => api.GenerationDefaults.audio(emptySampling()), {kind: 'Audio', parameters: {}}],
];
for (const raw of [' { "nested" : [true, null, {"x":7}], "label": "line\\ntext" } ',
    '[1, 2, null]', '42', 'true', '"text"', 'null']) {
    profiles.push(['other', () => api.GenerationDefaults.other(raw), {kind: 'Other', parameters: JSON.parse(raw)}]);
}
for (const [, makeDefaults, expected] of profiles) {
    for (const typed of [false, true]) {
        const parts = new api.ModelParts(read('model.gguf'));
        const defaults = makeDefaults();
        if (defaults !== undefined) parts.generation_defaults = defaults;
        const loader = new api.ModelLoader(api.ModelSource.parts(parts), config());
        const handle = typed ? undefined : loader.build();
        const model = typed ? loader.buildGenerative() : handle.asGenerative();
        consumed(loader);
        const observed = api.defaults_for_probe(model);
        const session = newSession(model);
        loader.free();
        handle?.free();
        model.free();
        assert.deepEqual(JSON.parse(observed.toJson()), expected);
        observed.free();
        assert.deepEqual(runSession(session), {tokens: [0, 1, 0], position: 5});
    }
}
cases.push(...new Set(profiles.map(([name]) => `parts-defaults-${name}`)));
for (const raw of ['', '{', 'null trailing', '{"x":NaN}']) {
    for (const typed of [false, true]) {
        const parts = new api.ModelParts(read('model.gguf'));
        parts.generation_defaults = api.GenerationDefaults.other(raw);
        const loader = new api.ModelLoader(api.ModelSource.parts(parts), config());
        assert.throws(() => typed ? loader.buildGenerative() : loader.build(), error =>
            error.code === 'InvalidConfig' && error.field === 'generation_defaults.raw_json'
            && error.value === raw && error.reason === 'invalid_json' && !!error.message);
        consumed(loader);
        loader.free();
    }
}
cases.push('parts-defaults-invalid-json');

for (const [arch, kind] of Object.entries({bert: 'Encoder', modernbert: 'Encoder', whisper: 'Whisper', silero_vad: 'Vad', kws: 'Hotword'})) {
    for (const typed of [false, true]) {
        const bad = new api.ModelLoader(api.ModelSource.bytes(read(`${arch}.gguf`)), config('metal'));
        assert.throws(() => typed ? bad.buildGenerative() : bad.build(), error =>
            error.code === 'KindMismatch' && error.expected === 'Generative' && error.actual === kind && error.architecture === arch,
            `${arch} kind mismatch payload (typed=${typed})`);
        consumed(bad);
        bad.free();
    }
    cases.push(`kind-${arch}`);
}
for (const name of ['unknown', 'malformed', 'backend', 'invalid-backend', 'assembly', 'inference']) {
    for (const typed of [false, true]) {
        const bytes = name === 'unknown' ? read('unknown.gguf') : name === 'assembly' ? read('llama.gguf') : name === 'malformed' ? new Uint8Array([0, 1]) : read('model.gguf');
        const backend = name === 'backend' ? 'metal' : name === 'invalid-backend' ? 'invalid-probe' : 'cpu';
        let source;
        if (name === 'inference') {
            const parts = new api.ModelParts(bytes);
            parts.inference_type = 'future/unsupported';
            source = api.ModelSource.parts(parts);
        } else {
            source = api.ModelSource.bytes(bytes);
        }
        const bad = new api.ModelLoader(source, config(backend));
        assert.throws(() => typed ? bad.buildGenerative() : bad.build(), error => {
            if (name === 'unknown') return error.code === 'UnsupportedArchitecture' && error.architecture === 'future_probe';
            if (name === 'inference') return error.code === 'UnsupportedInferenceType' && error.inference_type === 'future/unsupported';
            if (!error.message) return false;
            if (name === 'malformed') return error.code === 'Source' && error.source_kind === 'bytes';
            if (name === 'backend' || name === 'assembly') return error.code === 'Assembly' && error.backend === (name === 'backend' ? 'Metal' : 'Cpu');
            return name === 'invalid-backend' && error.code === 'InvalidConfig' && error.field === 'backend'
                && error.value === 'invalid-probe' && error.reason === 'unknown_backend';
        });
        consumed(bad);
        bad.free();
    }
    cases.push(name);
}
const future = api.future_handle_for_probe();
assert.equal(future.kind(), 'future-probe');
assert.equal(future.asGenerative(), undefined);
future.free();
cases.push('future-kind');
cases.push(...require('./production_probe.cjs')(api, read('model.gguf')));
console.log(JSON.stringify({cases: cases.sort(), generation}));
