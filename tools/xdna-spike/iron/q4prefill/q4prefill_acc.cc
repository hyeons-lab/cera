#include <stdint.h>

extern "C" {
void q4_prefill_acc_impl(
    const uint8_t *__restrict x_slot,
    const uint8_t *__restrict w_tile,
    int32_t tile_idx,
    int32_t add_prev);

void q4_prefill_acc(
    const uint8_t *__restrict x_slot,
    const uint8_t *__restrict w_tile,
    int32_t tile_idx,
    int32_t add_prev) {
  q4_prefill_acc_impl(x_slot, w_tile, tile_idx, add_prev);
}
}
