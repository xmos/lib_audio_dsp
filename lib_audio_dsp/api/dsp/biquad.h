
#pragma once

#include "xmath/types.h"

int32_t adsp_biquad(
  int32_t new_sample,
  q2_30 coeffs[5],
  int32_t state[8],
  left_shift_t lsh);

void adsp_design_biquad_bypass(q2_30 coeffs[5]);

void adsp_design_biquad_mute(q2_30 coeffs[5]);

left_shift_t adsp_design_biquad_gain(q2_30 coeffs[5], const float gain_db);

void adsp_design_biquad_lowpass(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q);

void adsp_design_biquad_highpass(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q);

void adsp_design_biquad_bandpass(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float bandwidth);

void adsp_design_biquad_bandstop(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float bandwidth);

void adsp_design_biquad_notch(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q);

void adsp_design_biquad_allpass(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q);

left_shift_t adsp_design_biquad_peaking(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db);

left_shift_t adsp_design_biquad_const_q(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db);
  
left_shift_t adsp_design_biquad_lowshelf(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db);
  
left_shift_t adsp_design_biquad_highshelf(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db);

void adsp_design_biquad_linkwitz(
  q2_30 coeffs[5],
  const float f0,
  const float fs,
  const float q0,
  const float fp,
  const float qp);  
