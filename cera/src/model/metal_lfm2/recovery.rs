use super::MetalLfm2Model;
use crate::kv_cache::{InferenceState, KvCompression};
use crate::model::Model;
use crate::session::CeraError;
use std::sync::atomic::Ordering;

impl MetalLfm2Model {
    pub(super) fn reset_kv_checked(
        &self,
        state: &mut InferenceState,
        compression: &KvCompression,
        max_seq_len: usize,
    ) -> Result<(), CeraError> {
        let mut fresh = InferenceState::from_config_capped(&self.config, compression, max_seq_len)?;
        fresh.lora = state.lora.clone();
        // Configuration locks internally. The mode is immutable once selected;
        // acquire the operation lock afterward to avoid recursive locking.
        self.configure_kv_compression(compression)?;
        let _guard = self.infer_lock.lock().unwrap_or_else(|e| e.into_inner());

        // A previous operation may have unwound after submission. Fence the
        // queue before writing shared CPU/GPU convolution memory.
        let fence = self.ctx.queue.new_command_buffer();
        fence.set_label("checked-kv-reset-fence");
        fence.commit();
        fence.wait_until_completed();
        if fence.status() != metal::MTLCommandBufferStatus::Completed {
            return Err(CeraError::Backend(format!(
                "checked Metal reset fence failed: {:?}",
                fence.status()
            )));
        }
        self.zero_conv_buffers_locked();
        // Attention kernels read only live positions; both ordinary and
        // compressed tails become inaccessible when the device length is zero.
        self.state.seq_len.store(0, Ordering::Relaxed);
        *state = fresh;
        Ok(())
    }
}
