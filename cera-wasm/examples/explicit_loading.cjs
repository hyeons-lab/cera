const fs = require('node:fs');
const path = require('node:path');

// Arguments: generated Node module, generative GGUF, raw completion prompt.
if (process.argv.length !== 5) {
    throw new Error('usage: node explicit_loading.cjs cera_wasm.js model.gguf prompt');
}
const api = require(path.resolve(process.argv[2]));
const bytes = new Uint8Array(fs.readFileSync(process.argv[3]));
const owned = [];
const keep = value => { owned.push(value); return value; };
try {
    // Source and LoadConfig transfer ownership into the loader.
    const loader = keep(new api.ModelLoader(api.ModelSource.bytes(bytes), new api.LoadConfig(4096, 'cpu')));
    const model = keep(loader.buildGenerative());
    const engine = keep(model.engine());
    const config = keep(new api.SessionConfig());
    config.seed = 42n;
    const session = keep(model.createSession(config));
    const tokenizer = keep(engine.tokenizer);
    session.appendTokens(tokenizer.encode(process.argv[4]));
    const options = keep(new api.GenerateOpts());
    Object.assign(options, {maxTokens: 32, temperature: 0.7});
    const tokens = [];
    keep(session.generate(options, batch => tokens.push(...batch)));
    console.log(tokenizer.decode(new Uint32Array(tokens)));
} finally {
    for (const value of owned.reverse()) value.free();
}
