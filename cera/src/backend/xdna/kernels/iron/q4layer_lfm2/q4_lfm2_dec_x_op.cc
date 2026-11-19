#include <stdint.h>

extern "C" {
void q4_lfm2_dec_x_op_impl(
    const float *__restrict x_pkt,
    int32_t op,
    int32_t a0,
    int32_t a1,
    int32_t a2);

void q4_lfm2_dec_x_op(
    const float *__restrict x_pkt,
    int32_t op,
    int32_t a0,
    int32_t a1,
    int32_t a2) {
  q4_lfm2_dec_x_op_impl(x_pkt, op, a0, a1, a2);
}
}
