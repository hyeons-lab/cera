#include <stdint.h>

extern "C" {
void q4_lfm3_conv_y_op_impl(float *__restrict y_out, int32_t op, int32_t arg);

void q4_lfm3_conv_y_op(float *__restrict y_out, int32_t op, int32_t arg) {
  q4_lfm3_conv_y_op_impl(y_out, op, arg);
}
}
