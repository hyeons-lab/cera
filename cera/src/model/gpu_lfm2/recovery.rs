use super::GpuLfm2Model;
use crate::kv_cache::{InferenceState, KvCompression};
use crate::model::Model;
use crate::session::CeraError;
use std::sync::atomic::Ordering;

impl GpuLfm2Model {
    pub(super) fn reset_kv_checked(
        &self,
        state: &mut InferenceState,
        compression: &KvCompression,
        max_seq_len: usize,
    ) -> Result<(), CeraError> {
        let mut fresh = InferenceState::from_config_capped(&self.config, compression, max_seq_len)?;
        fresh.lora = state.lora.clone();
        self.configure_kv_compression(compression)?;
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.zero_conv_buffers_locked();

        // A strict four-byte readback fences the clear and earlier submissions.
        // download_f32 cannot establish success: it returns zero-filled data on
        // mapping failure. This native-only path uses the existing fallible map
        // completion instead; browser callers need asynchronous recovery.
        let completion = self.ctx.begin_download(&self.logits_buf, 4);
        let bytes = pollster::block_on(completion.recv())
            .map_err(|error| CeraError::Backend(format!("checked wgpu reset: {error:#}")))?;
        if bytes.len() != 4 {
            return Err(CeraError::Backend(
                "checked wgpu reset returned an incomplete completion readback".into(),
            ));
        }
        self.gpu_state.seq_len.store(0, Ordering::Relaxed);
        *state = fresh;
        Ok(())
    }
}
