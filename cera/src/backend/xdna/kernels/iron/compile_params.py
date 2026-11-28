"""Pure compile-time parameter validation helpers for XDNA IRON kernel generators."""

from __future__ import annotations

from collections.abc import Collection


def validate_compile_params(
    layer_mode: int,
    kv_chunks: int,
    *,
    valid_modes: Collection[int],
    nonzero_kv_modes: Collection[int] = (),
    max_kv_chunks: int = 256,
    param_slot: int | None = None,
    valid_param_slots: Collection[int] | None = None,
) -> None:
    """Validate compile-time parameters before lowering or CLI dispatch."""
    modes_str = ", ".join(str(m) for m in valid_modes)
    if isinstance(layer_mode, bool) or not isinstance(layer_mode, int):
        raise ValueError(
            f"unsupported layer_mode {layer_mode!r}; expected one of {modes_str}"
        )
    if layer_mode not in valid_modes:
        raise ValueError(
            f"unsupported layer_mode {layer_mode}; expected one of {modes_str}"
        )
    if isinstance(kv_chunks, bool) or not isinstance(kv_chunks, int):
        raise ValueError(
            f"kv_chunks ({kv_chunks!r}) out of range 0..={max_kv_chunks}"
        )
    if not (0 <= kv_chunks <= max_kv_chunks):
        raise ValueError(
            f"kv_chunks ({kv_chunks}) out of range 0..={max_kv_chunks}"
        )
    if layer_mode in nonzero_kv_modes and kv_chunks == 0:
        raise ValueError(
            f"layer_mode={layer_mode} requires kv_chunks >= 1"
        )
    if param_slot is not None and valid_param_slots is None:
        raise ValueError("valid_param_slots is required when param_slot is provided")
    if valid_param_slots is not None:
        slots_str = ", ".join(str(s) for s in valid_param_slots)
        if isinstance(param_slot, bool) or not isinstance(param_slot, int):
            raise ValueError(
                f"unsupported param_slot {param_slot!r}; expected one of {slots_str}"
            )
        if param_slot not in valid_param_slots:
            raise ValueError(
                f"unsupported param_slot {param_slot}; expected one of {slots_str}"
            )
