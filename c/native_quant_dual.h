/*
 * Dual reference matvecs: two weight matrices (for example an expert's gate and
 * up projections) applied to the same input in one pass, so the input is read
 * and quantized once. FP4 and FP8 variants.
 */

#ifndef COLIBRI_NATIVE_QUANT_DUAL_H
#define COLIBRI_NATIVE_QUANT_DUAL_H

#include "tensor.h"

int coli_fp4_dual_matvec_ref(float *output_a, float *output_b,
                             const ColiTensorView *weight_a,
                             const ColiTensorView *weight_b,
                             const float *input);
int coli_fp8_dual_matvec_ref(float *output_a, float *output_b,
                             const ColiTensorView *weight_a,
                             const ColiTensorView *weight_b,
                             const float *input);

#endif
