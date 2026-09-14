// CUDA PTX module loading, caching, and kernel function management.
//
// Loads compiled PTX bytecode directly into the CUDA driver JIT.
// Caches compiled modules and exposes entry-point kernel handles for execution.

use std::sync::Arc;

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaFunction as RawCudaFunction, CudaModule as RawCudaModule};
use cudarc::nvrtc::Ptx;

/// Compiled CUDA module loaded in a device context.
#[derive(Debug, Clone)]
pub struct CudaModule {
    pub(crate) module: Arc<RawCudaModule>,
}

impl CudaModule {
    /// Load a compiled PTX assembly string into the CUDA device context.
    pub fn from_ptx(ctx: &Arc<CudaContext>, ptx_src: &str) -> Result<Self> {
        let ptx = Ptx::from_src(ptx_src);
        let module = ctx
            .load_module(ptx)
            .context("failed to load PTX module into CUDA context")?;
        Ok(Self { module })
    }

    /// Compile CUDA C++ source via NVRTC and load it into the CUDA device context.
    pub fn from_cuda_src(
        ctx: &Arc<CudaContext>,
        cu_src: &str,
        name: Option<&str>,
        arch: Option<&'static str>,
    ) -> Result<Self> {
        let opts = cudarc::nvrtc::CompileOptions {
            arch,
            use_fast_math: Some(true),
            name: name.map(|s| s.to_string()),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(cu_src, opts).map_err(|e| {
            anyhow::anyhow!(
                "NVRTC compilation failed for {}: {e:?}",
                name.unwrap_or("kernel")
            )
        })?;
        let module = ctx
            .load_module(ptx)
            .context("failed to load compiled PTX module into CUDA context")?;
        Ok(Self { module })
    }

    /// Retrieve an entry-point kernel function by name.
    pub fn get_kernel(&self, name: &str) -> Result<CudaKernel> {
        let func = self
            .module
            .load_function(name)
            .with_context(|| format!("failed to load CUDA function '{name}' from module"))?;
        Ok(CudaKernel { func })
    }

    /// Underlying raw cudarc module reference.
    pub fn inner(&self) -> &Arc<RawCudaModule> {
        &self.module
    }
}

/// Handle to a callable CUDA device kernel function.
#[derive(Debug, Clone)]
pub struct CudaKernel {
    pub(crate) func: RawCudaFunction,
}

impl CudaKernel {
    /// Underlying raw cudarc function reference.
    pub fn inner(&self) -> &RawCudaFunction {
        &self.func
    }
}
