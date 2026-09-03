# cera_ffi_flutter_example

A chat app built on `cera_ffi_flutter`: pick a GGUF, talk to it, watch tokens
stream in. One code path serves Android, iOS, macOS, Linux, Windows and the
browser, because it is written against `Cera`, the portable async API, rather
than against the generated `dart:ffi` bindings.

```bash
flutter run                # whichever device is attached
```

## Running it in the browser

The web build needs the wasm runtime installed into `web/` once. It is not part
of the package archive: it is a build output of the Rust crate, and this package
follows the same rule as every other target here, where binaries come from a
release rather than from git.

```bash
just wasm-web-wgpu                                          # from the repo root
cd cera_ffi_flutter/example
dart run cera_ffi_flutter:install_web \
  --from ../../cera-wasm/examples/webgpu/pkg --force
flutter run -d chrome
```

`--force` matters on every run after the first: without it the tool skips files
that already exist, so a `web/cera/` left over from an earlier install stays
exactly as it was.

**Build it with `just wasm-web-wgpu`, not `just wasm-web`.** Only the former
passes `--features wgpu`, and that feature is the whole point of running in a
browser: it is the difference between roughly 58 tok/s on WebGPU and roughly
1.4 tok/s on the wasm CPU path. Getting it wrong is quiet. `cera_worker.js`
probes for `WebGpuSession` and falls through to the CPU when the wasm does not
carry it, so the wrong build produces a working demo rather than an error.

Two things decide whether you are actually on the GPU:

- **The model architecture must be supported on WebGPU.** `WebGpuSession` supports
  `lfm2`/`lfm2.5`/`lfm2moe` as well as dense transformers (`llama`, `qwen2`, `qwen3`,
  `granite`, with classic Mistral served under `llama`).
- **The browser must expose `navigator.gpu`.** WebGPU is allowed on `localhost`
  without HTTPS, so local development needs no certificate.

## The hosted demo

Pushes to `main` deploy this app to Cloudflare Pages, and pull requests opened
from this repository get their own preview URL. Both are path-filtered to the
trees the build actually consumes, so a change confined to `docs/` or the root
README deploys nothing (a change to this file does deploy, since it sits inside
one of them). PRs from forks are skipped, because they cannot read the deploy
secrets. See
`.github/workflows/deploy-web-demo.yml`, whose header documents the one-time
Cloudflare setup and how to attach a custom domain.

Nothing about the deployment hosts a model. The app reads the GGUF the visitor
picks off their own disk, which is what keeps it inside Cloudflare's 25 MiB
per-file limit and off any bandwidth bill. If you ever want a model preloaded,
it belongs in R2 with CORS enabled, not in the Pages deployment.
