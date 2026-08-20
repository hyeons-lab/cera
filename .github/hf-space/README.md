---
title: Cera WebGPU Demo
emoji: ⚡
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
header: mini
fullWidth: true
license: other
license_name: apache-2.0-or-mit
license_link: https://github.com/hyeons-lab/cera#license
short_description: LFM2 running on WebGPU, entirely in your own browser
tags:
  - webgpu
  - webassembly
  - on-device
  - lfm2
models:
  - LiquidAI/LFM2-350M-GGUF
  - LiquidAI/LFM2-1.2B-GGUF
---

# Cera on WebGPU

[Cera](https://github.com/hyeons-lab/cera) is an inference engine written in
Rust. This page is its Flutter example app, running LFM2 on your GPU through
WebGPU: the engine is compiled to WebAssembly, and the app around it is the
usual Flutter web build.

Nothing is uploaded and nothing is generated on a server. The model you choose
is read in the browser tab you are looking at, and the tokens come back from
your own GPU. Closing the tab is the whole cleanup story.

## Using it

1. Open it in a browser with WebGPU. Chrome, Edge and Safari have shipped it;
   recent Firefox has it on some platforms and behind `dom.webgpu.enabled` on
   others. If the page reports the WebAssembly CPU backend, that is what
   happened.
2. Download an **LFM2** GGUF, for example
   [`LFM2-350M-GGUF`](https://huggingface.co/LiquidAI/LFM2-350M-GGUF) (small
   and quick to fetch) or
   [`LFM2-1.2B-GGUF`](https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF).
3. Pick the file, and chat.

The architecture matters: the WebGPU path implements LFM2, so any other
architecture falls back to the WebAssembly CPU path and runs roughly 40x slower
(around 1.4 tok/s against around 58). The status line names the backend it
ended up on.

Picking a file off your own disk is a rough edge, not the design. The engine
can already resolve a model by bundle id straight from the Hub, though only on
the WebAssembly CPU path today, so fetching one for the WebGPU session is still
work rather than a switch to flip.

## About this repo

Generated, not authored. The site is the output of `flutter build web` in
[hyeons-lab/cera](https://github.com/hyeons-lab/cera), and this card is a file
in that repo copied in beside it. The `Deploy Web Demo` workflow force-pushes
the lot. Edits made in the Hub UI are discarded by the next publish, so send
changes to the source repo instead.

Cera is dual-licensed Apache-2.0 or MIT, at your option.
