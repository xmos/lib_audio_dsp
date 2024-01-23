
#pragma once

int32_t adsp_biquad(
  int32_t new_sample,
  int32_t coeffs[5],
  int32_t state[8],
  left_shift_t lsh);

void adsp_design_biquad_bypass(int32_t coeffs[5]);

void adsp_design_biquad_mute(int32_t coeffs[5]);

void adsp_design_biquad_lowpass(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q);

void adsp_design_biquad_highpass(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q);

void adsp_design_biquad_bandpass(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float bandwidth);

void adsp_design_biquad_bandstop(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float bandwidth);

void adsp_design_biquad_notch(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q);

void adsp_design_biquad_allpass(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q);

left_shift_t adsp_design_biquad_peaking(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db);

left_shift_t adsp_design_biquad_const_q(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db);
  
left_shift_t adsp_design_biquad_lowshelf(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db);
  
left_shift_t adsp_design_biquad_highshelf(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db);

void adsp_design_biquad_linkwitz(
  int32_t coeffs[5],
  const float f0,
  const float fs,
  const float q0,
  const float fp,
  const float qp);  
