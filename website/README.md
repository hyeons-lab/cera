# Cera Official Website & WebGPU Studio

The official website and interactive documentation portal for [Cera](https://github.com/hyeons-lab/cera), a high-performance, pure-Rust on-device AI inference engine.

## Features

- **Multiplatform SDK Documentation**: Detailed guides and code snippets for Rust (`cera`), Web/JS (`cera-wasm`), Flutter/Dart (`cera_ffi_flutter`), Android/Kotlin (`cera-ffi-kotlin`), Apple/Swift (`Package.swift`), and Python.
- **WebGPU Inference Studio**: In-browser local model runner with live token telemetry, thought/reasoning visualization, and TurboQuant KV compression toggles.
- **LeapBundles Catalog**: Interactive browser for published GGUF models with one-click CLI copy commands.
- **Hardware Benchmarks**: Comprehensive performance matrix across Apple Silicon (Metal MSL), PC/Mobile GPUs (WebGPU/Vulkan/DX12), and CPU SIMD (NEON/AVX-512).

## Tech Stack

- **Framework**: React 19 + TypeScript
- **Bundler**: Vite
- **Styling**: Tailwind CSS v4
- **Icons**: Lucide React
- **Hosting**: Cloudflare Pages (`cera-site`)

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build production bundle (tsc typecheck + vite build)
npm run build

# Preview production build
npm run preview
```
