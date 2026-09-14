// Shared probe conversion for native and wasm32 boundary execution.
pub fn native_context_size(context_size: u64) -> Result<usize, String> {
    if context_size == 0 {
        Ok(usize::MAX)
    } else {
        usize::try_from(context_size)
            .map_err(|_| format!("context_size {context_size} exceeds usize::MAX on this target"))
    }
}
