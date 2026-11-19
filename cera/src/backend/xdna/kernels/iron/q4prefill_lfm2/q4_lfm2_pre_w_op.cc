#include "q4_lfm2_fa_op.cc"

extern "C" {
void q4_lfm2_pre_w_op_impl(const float *__restrict w_pkt, int32_t op, int32_t a0, int32_t a1);

void q4_lfm2_pre_w_op(const float *__restrict w_pkt, int32_t op, int32_t a0, int32_t a1) {
  q4_lfm2_pre_w_op_impl(w_pkt, op, a0, a1);
}
}
