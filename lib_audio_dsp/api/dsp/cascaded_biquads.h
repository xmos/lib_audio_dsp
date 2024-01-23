
#pragma once

int32_t adsp_cascaded_biquads_8b(
  int32_t new_sample,
  int32_t coeffs[40],
  int32_t state[64],
  left_shift_t lsh[8]);

void adsp_design_butterworth_lowpass_8b(
  int32_t coeffs[40],
  const unsigned N,
  const float fc,
  const float fs);

void adsp_design_butterworth_highpass_8b(
  int32_t coeffs[40],
  const unsigned N,
  const float fc,
  const float fs);
