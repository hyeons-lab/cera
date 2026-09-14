const assert = require('node:assert/strict');

function snapshot(config) {
    const kv = config.kvCompression;
    const result = {
        maxSeqLen: config.maxSeqLen, nKeep: config.nKeep,
        seed: config.seed, ubatchSize: config.ubatchSize,
        kv: kv && {seed: kv.seed, keys: kv.keys, values: kv.values},
    };
    kv?.free();
    return result;
}

module.exports = function runProduction(api, bytes) {
    const defaults = new api.SessionConfig();
    assert.deepEqual(snapshot(defaults), {
        maxSeqLen: undefined, nKeep: 0, seed: undefined, ubatchSize: 512, kv: undefined,
    });
    for (const multipart of [false, true]) {
        for (const typed of [false, true]) {
            const source = multipart
                ? api.ModelSource.parts(new api.ModelParts(bytes)) : api.ModelSource.bytes(bytes);
            const loader = new api.ModelLoader(source, new api.LoadConfig(24, 'cpu'));
            const handle = typed ? undefined : loader.build();
            const model = typed ? loader.buildGenerative() : handle.asGenerative();
            const engine = model.engine();
            assert.ok(engine instanceof api.CeraEngine);
            assert.ok(api.engines_share_for_probe(model, engine));
            const independent = api.CeraEngine.fromGgufBytes(bytes, 24);
            assert.equal(api.engines_share_for_probe(model, independent), false);
            independent.free();
            assert.equal(engine.contextSize, 24);
            assert.equal(engine.maxSeqLen, 24);
            const config = new api.SessionConfig();
            Object.assign(config, {maxSeqLen: 8, seed: 0xffffffffffffffffn, ubatchSize: 1});
            const session = model.createSession(config);
            assert.ok(session instanceof api.Session);
            const observed = api.session_config_for_probe(session);
            assert.deepEqual(snapshot(observed), snapshot(config));
            observed.free();
            config.free();
            const sibling = engine.newSession(defaults);
            loader.free();
            handle?.free();
            model.free();
            assert.equal(engine.contextSize, 24);
            const tokenizer = engine.tokenizer;
            engine.free();
            assert.deepEqual(Array.from(tokenizer.encode('ab')), [0, 1]);
            assert.equal(tokenizer.decode(new Uint32Array([0, 1])), 'ab');
            tokenizer.free();
            sibling.appendTokens(new Uint32Array([1]));
            session.appendTokens(new Uint32Array([0, 1]));
            const opts = new api.GenerateOpts();
            Object.assign(opts, {maxTokens: 1, temperature: 0.7, ignoreEos: true, flushEveryTokens: 1});
            const tokens = [];
            const first = session.generate(opts, batch => tokens.push(...batch));
            assert.equal(first.tokensGenerated, 1);
            assert.equal(first.finishReason, 'MaxTokens');
            first.free();
            assert.equal(session.position, 3);
            opts.maxTokens = 2;
            const second = session.generate(opts, batch => tokens.push(...batch));
            assert.equal(second.tokensGenerated, 2);
            second.free();
            assert.equal(tokens.length, 3);
            assert.equal(session.position, 5);
            assert.equal(sibling.position, 1);
            session.cancel();
            const cancelled = session.generate(opts, () => assert.fail('cancelled session emitted tokens'));
            assert.equal(cancelled.finishReason, 'Cancelled');
            assert.equal(cancelled.tokensGenerated, 0);
            cancelled.free();
            assert.equal(session.position, 5);
            session.clearCancel();
            opts.maxTokens = 1;
            session.generate(opts, () => {}).free();
            assert.equal(session.position, 6);
            session.reset();
            assert.equal(session.position, 0);
            assert.throws(() => session.appendTokens(new Uint32Array()), /empty/i);
            opts.free();
            session.free();
            sibling.free();
        }
    }
    defaults.free();

    const loader = new api.ModelLoader(api.ModelSource.bytes(bytes), new api.LoadConfig(24, 'cpu'));
    const model = loader.buildGenerative();
    loader.free();
    for (const mode of [undefined, [0n, true, true], [0xffffffffffffffffn, true, false], [42n, false, true]]) {
        const config = new api.SessionConfig();
        Object.assign(config, {maxSeqLen: 8, nKeep: 1, seed: 0n, ubatchSize: 0});
        if (mode) {
            const kv = new api.TurboQuantConfig(mode[0]);
            Object.assign(kv, {keys: mode[1], values: mode[2]});
            config.kvCompression = kv; // Assignment consumes this handle.
        }
        const session = model.createSession(config);
        const observed = api.session_config_for_probe(session);
        assert.deepEqual(snapshot(observed), snapshot(config));
        observed.free();
        config.free();
        session.appendTokens(new Uint32Array([0, 1]));
        const opts = new api.GenerateOpts();
        Object.assign(opts, {maxTokens: 1, temperature: 0, ignoreEos: true});
        const summary = session.generate(opts, () => {});
        assert.equal(summary.tokensGenerated, 1);
        summary.free();
        opts.free();
        session.free();
    }
    const cap = new api.SessionConfig();
    cap.maxSeqLen = 1;
    const capped = model.createSession(cap);
    cap.free();
    model.free();
    capped.appendTokens(new Uint32Array([0]));
    assert.throws(() => capped.appendTokens(new Uint32Array([1])), /context/i);
    assert.equal(capped.position, 1);
    capped.free();
    return ['production-defaults', 'production-session', 'production-shared-engine',
        'production-kv-config', 'production-stream-cancel'];
};
