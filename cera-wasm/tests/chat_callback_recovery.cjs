// node chat_callback_recovery.cjs /path/to/node-package /path/to/model.gguf
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const cera = require(path.join(path.resolve(process.argv[2]), 'cera_wasm.js'));
const engine = cera.CeraEngine.fromGgufBytes(fs.readFileSync(process.argv[3]), 128);
const config = new cera.SessionConfig(42n);
const chat = engine.newChatSession(config);
const opts = new cera.GenerateOpts();
opts.maxTokens = 1;
opts.temperature = 0;
const cyclic = [];
cyclic.push(cyclic);
const hostile = { get [Symbol.toStringTag]() { throw new Error('getter invoked'); } };

try {
  for (const json of [false, true]) {
    for (const value of [new Error('consumer failed'), cyclic, hostile]) {
      chat.ingest({role: 'user', content: 'Hi'});
      let calls = 0;
      const callback = () => { calls++; throw value; };
      assert.throws(() => json
        ? chat.generateStreamingJson(opts, '{"type":"string"}', callback)
        : chat.generateStreaming(opts, callback), /stream callback failed/);
      assert.equal(calls, 1);
      assert.equal(chat.phase, 'Interrupted');
      chat.reset();
      assert.equal(chat.phase, 'Idle');
    }
  }
  console.log('PASS: ordinary and JSON callback errors release the handle for all three thrown values');
} finally {
  chat.free();
  opts.free();
  config.free();
  engine.free();
}
