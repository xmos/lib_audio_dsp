// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "control/adsp_control.h"

#include <math.h>
#include <string.h>

#include <xcore/assert.h>
#include "control/helpers.h"

#define Q_factor 30

static const float pi =    (float)M_PI;
static const float log_2 = 0.69314718055f;

static inline float _check_fc(float fc, float fs) {
  float fc_sat = fc;
  // saturate if > fs/2
  if (fc_sat >= fs / 2.0f){
    fc_sat = fs / 2.0f;
  }
  return fc_sat;
}


static inline left_shift_t _get_b_shift(float b0, float b1, float b2) {

  // calculate the required headroom for the b coefficients

  float max_b = fabsf(b0);
  float tmp = fabsf(b1);
  if (tmp > max_b){
    max_b = tmp;
  }
  tmp = fabsf(b2);
  if (tmp > max_b){
    max_b = tmp;
  }

  if (max_b == 0){
    return 0;
  }

  tmp = floorf(log2f(max_b));
  left_shift_t out = (left_shift_t)tmp;

  return out > 0 ? out : 0;
}



biquad_slew_t adsp_biquad_slew_init(
  q2_30 target_coeffs[8],
  left_shift_t lsh,
  left_shift_t slew_shift
){
  biquad_slew_t slew_state;
  memcpy(slew_state.target_coeffs, target_coeffs, 5*sizeof(int32_t));
  memcpy(slew_state.active_coeffs, target_coeffs, 5*sizeof(int32_t));
  slew_state.remaining_shifts = 0;
  slew_state.lsh = lsh;
  slew_state.slew_shift = slew_shift < 1 ? 1 : slew_shift;
  return slew_state;
  }


void adsp_biquad_slew_update_coeffs(
  biquad_slew_t* slew_state,
  int32_t** states,
  int32_t channels,
  q2_30 target_coeffs[8],
  left_shift_t lsh
){
  left_shift_t old_shift = slew_state->lsh;
  slew_state->lsh = lsh;
  memcpy(slew_state->target_coeffs, target_coeffs, 5*sizeof(int32_t));

  left_shift_t b_shift_change = old_shift - slew_state->lsh;

  if (b_shift_change == 0){
    return;
  }
  else if(b_shift_change < 0){
    // we can shift down safely as we are increasing headroom
    b_shift_change = -b_shift_change;
    for (int i=0; i < 3; i++){
      slew_state->active_coeffs[i] >>= b_shift_change;
    }
    for (int i=0; i < channels; i++){
      states[i][3] >>= b_shift_change;
      states[i][4] >>= b_shift_change;
    }
    return;
  }
  else {
    // we can't shift safely until we know we have headroom
    slew_state->remaining_shifts = b_shift_change;
    slew_state->lsh += slew_state->remaining_shifts;
  }

}


left_shift_t adsp_design_biquad_bypass(q2_30 coeffs[5]) {
  coeffs[0] = 1 << Q_factor;
  coeffs[1] = 0;
  coeffs[2] = 0;
  coeffs[3] = 0;
  coeffs[4] = 0;

  // b_shift is always zero for this type of filter
  return 0;
}

left_shift_t adsp_design_biquad_mute(q2_30 coeffs[5]) {
  coeffs[0] = 0;
  coeffs[1] = 0;
  coeffs[2] = 0;
  coeffs[3] = 0;
  coeffs[4] = 0;

  // b_shift is always zero for this type of filter
  return 0;
}

left_shift_t adsp_design_biquad_gain(q2_30 coeffs[5], const float gain_db) {
  float A  = powf(10.0f, (gain_db * (1.0f / 20.0f)));

  left_shift_t b_sh = _get_b_shift(A, 0, 0);

  coeffs[0] = _float2fixed_assert( A, Q_factor - b_sh );
  coeffs[1] = 0;
  coeffs[2] = 0;
  coeffs[3] = 0;
  coeffs[4] = 0;

  return b_sh;
}


left_shift_t adsp_design_biquad_lowpass
(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q
) {
  float fc_sat = fc;
  // saturate if > fs/2
  if (fc_sat >= fs / 2.0f){
    fc_sat = fs / 2.0f;
  }
  
  // Compute common factors
  float K = tanf(pi * fc_sat/fs);
  float KK = K * K;
  float KQ = K / filter_Q;
  float norm = 1.0f / (1.0f + KQ + KK);
  
  // Compute coeffs
  float b0 = KK * norm;
  float b1 = 2.0f * b0;
  float b2 = b0;
  float a1 = 2.0f * (KK - 1.0f) * norm;
  float a2 = (1.0f - KQ + KK) * norm;

  // Store as fixed-point values
  coeffs[0] = _float2fixed_assert(  b0, Q_factor );
  coeffs[1] = _float2fixed_assert(  b1, Q_factor );
  coeffs[2] = _float2fixed_assert(  b2, Q_factor );
  coeffs[3] = _float2fixed_assert( -a1, Q_factor );
  coeffs[4] = _float2fixed_assert( -a2, Q_factor );

  // b_shift is always zero for this type of filter
  return 0;
}

left_shift_t adsp_design_biquad_highpass
(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q
) {
  float fc_sat = _check_fc(fc, fs);

  // Compute common factors
  float K = tanf(pi * fc_sat/fs);
  float KK = K * K;
  float KQ = K / filter_Q;
  float norm = 1.0f / (1.0f + KQ + KK);
  
  // Compute coeffs
  float b0 = norm;
  float b1 = -2.0f * b0;
  float b2 = b0;
  float a1 = 2.0f * (KK - 1.0f) * norm;
  float a2 = (1.0f - KQ + KK) * norm;

  // Store as fixed-point values
  coeffs[0] = _float2fixed_assert(  b0, Q_factor );
  coeffs[1] = _float2fixed_assert(  b1, Q_factor );
  coeffs[2] = _float2fixed_assert(  b2, Q_factor );
  coeffs[3] = _float2fixed_assert( -a1, Q_factor );
  coeffs[4] = _float2fixed_assert( -a2, Q_factor );

  // b_shift is always zero for this type of filter
  return 0;
}


left_shift_t adsp_design_biquad_bandpass
(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float bandwidth
) {
  float fc_sat = _check_fc(fc, fs);
  
  // Compute common factors
  float w0    = 2.0f * pi * fc_sat / fs;
  float sin_w0 = f32_sin(w0);
  float alpha = sin_w0 * sinhf(log_2 / 2.0f * bandwidth * w0 / sin_w0);

  // Compute coeffs
  float b0 =  alpha;
  float b1 =  0.0f;
  float b2 = -alpha;
  float a0 =  1.0f + alpha;
  float a1 = -2.0f * f32_cos(w0);
  float a2 =  1.0f - alpha;

  float inv_a0 = 1.0f/a0;

  // Store as fixed-point values
  coeffs[0] = _float2fixed_assert(  b0 * inv_a0, Q_factor );
  coeffs[1] = _float2fixed_assert(  b1 * inv_a0, Q_factor );
  coeffs[2] = _float2fixed_assert(  b2 * inv_a0, Q_factor );
  coeffs[3] = _float2fixed_assert( -a1 * inv_a0, Q_factor );
  coeffs[4] = _float2fixed_assert( -a2 * inv_a0, Q_factor );

  // b_shift is always zero for this type of filter
  return 0;
}

left_shift_t adsp_design_biquad_bandstop
(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float bandwidth
) {
  float fc_sat = _check_fc(fc, fs);

  // Compute common factors
  float w0    = 2.0f * pi * fc_sat / fs;
  float sin_w0 = f32_sin(w0);
  float alpha = sin_w0 * sinhf(log_2 / 2.0f * bandwidth * w0 / sin_w0);

  // Compute coeffs
  float b0 =  1.0f;
  float b1 = -2.0f * f32_cos(w0);
  float b2 =  1.0f;
  float a0 =  1.0f + alpha;
  float a1 =  b1;
  float a2 =  1.0f - alpha;

  float inv_a0 = 1.0f/a0;

  // Store as fixed-point values
  coeffs[0] = _float2fixed_assert(  b0 * inv_a0, Q_factor );
  coeffs[1] = _float2fixed_assert(  b1 * inv_a0, Q_factor );
  coeffs[2] = _float2fixed_assert(  b2 * inv_a0, Q_factor );
  coeffs[3] = _float2fixed_assert( -a1 * inv_a0, Q_factor );
  coeffs[4] = _float2fixed_assert( -a2 * inv_a0, Q_factor );

  // b_shift is always zero for this type of filter
  return 0;
}

left_shift_t adsp_design_biquad_notch
(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q
) {
  float fc_sat = _check_fc(fc, fs);
  
  // Compute common factors
  float K = tanf(pi * fc_sat/fs);
  float KK = K * K;
  float KQ = K / filter_Q;
  float norm = 1.0f / (1.0f + KQ + KK);
  
  // Compute coeffs
  float b0 = (1.0f + KK) * norm;
  float b1 = 2.0f * (KK - 1.0f) * norm;
  float b2 = b0;
  float a1 = b1;
  float a2 = (1.0f - KQ + KK) * norm;

  // Store as fixed-point values
  coeffs[0] = _float2fixed_assert(  b0, Q_factor );
  coeffs[1] = _float2fixed_assert(  b1, Q_factor );
  coeffs[2] = _float2fixed_assert(  b2, Q_factor );
  coeffs[3] = _float2fixed_assert( -a1, Q_factor );
  coeffs[4] = _float2fixed_assert( -a2, Q_factor );

  // b_shift is always zero for this type of filter
  return 0;
}


left_shift_t adsp_design_biquad_allpass
(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q
) {
  float fc_sat = _check_fc(fc, fs);

  // Compute common factors
  float K = tanf(pi * fc_sat/fs);
  float KK = K * K;
  float KQ = K / filter_Q;
  float norm = 1.0f / (1.0f + KQ + KK);
  
  // Compute coeffs
  float b0 = (1.0f - KQ + KK) * norm;
  float b1 = 2.0f * (KK - 1.0f) * norm;
  float b2 = 1.0f;
  float a1 = b1;
  float a2 = b0;

  // Store as fixed-point values
  coeffs[0] = _float2fixed_assert(  b0, Q_factor );
  coeffs[1] = _float2fixed_assert(  b1, Q_factor );
  coeffs[2] = _float2fixed_assert(  b2, Q_factor );
  coeffs[3] = _float2fixed_assert( -a1, Q_factor );
  coeffs[4] = _float2fixed_assert( -a2, Q_factor );

  // b_shift is always zero for this type of filter
  return 0;
}


left_shift_t adsp_design_biquad_peaking
(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db
) {
  float fc_sat = _check_fc(fc, fs);

  // Compute common factors
  float A  = powf(10.0f, (gain_db * (1.0f / 40.0f)));
  float w0 = 2.0f * pi * (fc_sat / fs); 
  // intentional double precision, gets extra precision
  float alpha = f32_sin(w0) / (2.0 * filter_Q);

  // Compute coeffs
  float norm = 1.0f /(1.0f + alpha / A);

  float b0 =  (1.0f + alpha * A)*norm;
  float b1 = (-2.0f * f32_cos(w0))*norm;
  float b2 =  (1.0f - alpha * A)*norm;
  float a1 =  b1;
  float a2 =  (1.0f - alpha / A)*norm;

  left_shift_t b_sh = _get_b_shift(b0, b1, b2);

  // Store as fixed-point values
  coeffs[0] = _float2fixed_assert(  b0, Q_factor - b_sh);
  coeffs[1] = _float2fixed_assert(  b1, Q_factor - b_sh);
  coeffs[2] = _float2fixed_assert(  b2, Q_factor - b_sh);
  coeffs[3] = _float2fixed_assert( -a1, Q_factor );
  coeffs[4] = _float2fixed_assert( -a2, Q_factor );

  return b_sh;
}

left_shift_t adsp_design_biquad_const_q
(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db
) {
  float fc_sat = _check_fc(fc, fs);

  // Compute common factors
  float V = powf(10.0f, (gain_db * (1.0f/ 20.0f)));
  // w0 is only needed for calculating K
  float K = tanf(pi * fc_sat / fs);

  float factor_a = K / filter_Q;
  float factor_b = 0;
  float K_pow2 = K * K;
  if(gain_db > 0) {
    factor_b = V * factor_a;
  }
  else
  {
    factor_b = factor_a;
    factor_a = factor_b / V;
  }

  // Compute coeffs
  float b0 = 1.0f + factor_b + K_pow2;
  float b1 = 2.0f * (K_pow2  - 1.0f);
  float b2 = 1.0f - factor_b + K_pow2;
  float a0 = 1.0f + factor_a + K_pow2;
  float a1 = b1;
  float a2 = 1.0f - factor_a + K_pow2;

  float inv_a0 = 1.0f/a0;

  b0 *= inv_a0;
  b1 *= inv_a0;
  b2 *= inv_a0;

  left_shift_t b_sh = _get_b_shift(b0, b1, b2);

  // Store as fixed-point values
  coeffs[0] = _float2fixed_assert(  b0, Q_factor - b_sh);
  coeffs[1] = _float2fixed_assert(  b1, Q_factor - b_sh);
  coeffs[2] = _float2fixed_assert(  b2, Q_factor - b_sh);
  coeffs[3] = _float2fixed_assert( -a1 * inv_a0, Q_factor );
  coeffs[4] = _float2fixed_assert( -a2 * inv_a0, Q_factor );

  return b_sh;
}

left_shift_t adsp_design_biquad_lowshelf
(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db
) {
  float fc_sat = _check_fc(fc, fs);

  // Compute common factors
  float A  = powf(10.0f, (gain_db * (1.0f / 40.0f)));
  float w0 = 2.0f * pi * fc_sat / fs;
  float alpha = sinf(w0) / (2.0f * filter_Q);

  float cosw0 = cosf(w0);
  float alpha_factor = 2.0f * sqrtf(A) * alpha;
  float Am1_cosw0 = (A - 1.0f) * cosw0;
  float Ap1_cosw0 = (A + 1.0f) * cosw0;

  // Compute coeffs
  float b0 =  A * ((A + 1.0f) - Am1_cosw0 + alpha_factor);
  float b1 =  2.0f * A * ((A - 1.0f) - Ap1_cosw0);
  float b2 =  A * ((A + 1.0f) - Am1_cosw0 - alpha_factor);
  float a0 = (A + 1.0f) + Am1_cosw0 + alpha_factor;
  float a1 = -2.0f * ((A - 1.0f) + Ap1_cosw0);
  float a2 = (A + 1.0f) + Am1_cosw0 - alpha_factor;

  float inv_a0 = 1.0f / a0;

  b0 *= inv_a0;
  b1 *= inv_a0;
  b2 *= inv_a0;

  left_shift_t b_sh = _get_b_shift(b0, b1, b2);

  // Store as fixed-point values
  coeffs[0] = _float2fixed_assert(  b0, Q_factor - b_sh);
  coeffs[1] = _float2fixed_assert(  b1, Q_factor - b_sh);
  coeffs[2] = _float2fixed_assert(  b2, Q_factor - b_sh);
  coeffs[3] = _float2fixed_assert( -a1 * inv_a0, Q_factor );
  coeffs[4] = _float2fixed_assert( -a2 * inv_a0, Q_factor );

  return b_sh;
}

left_shift_t adsp_design_biquad_highshelf
(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db
) {
  float fc_sat = _check_fc(fc, fs);

  // Compute common factors
  float A  = powf(10.0f, (gain_db * (1.0f / 40.0f)));
  float w0 = 2.0f * pi * fc_sat / fs;
  float alpha = sinf(w0) / (2.0f * filter_Q);

  float alpha_factor = 2.0f * sqrtf(A) * alpha;
  float cosw0 = cosf(w0);
  float Am1_cosw0 = (A - 1.0f) * cosw0;
  float Ap1_cosw0 = (A + 1.0f) * cosw0;

  // Compute coeffs
  float b0 =  A * ((A + 1.0f) + Am1_cosw0 + alpha_factor);
  float b1 = -2.0f * A * ((A - 1.0f) + Ap1_cosw0);
  float b2 =  A * ((A + 1.0f) + Am1_cosw0 - alpha_factor);
  float a0 = (A + 1.0f) - Am1_cosw0 + alpha_factor;
  float a1 =  2.0f * ((A - 1.0f) - Ap1_cosw0);
  float a2 = (A + 1.0f) - Am1_cosw0 - alpha_factor;

  float inv_a0 = 1.0f/a0;

  b0 *= inv_a0;
  b1 *= inv_a0;
  b2 *= inv_a0;

  left_shift_t b_sh = _get_b_shift(b0, b1, b2);

  // Store as fixed-point values
  coeffs[0] = _float2fixed_assert(  b0, Q_factor - b_sh);
  coeffs[1] = _float2fixed_assert(  b1, Q_factor - b_sh);
  coeffs[2] = _float2fixed_assert(  b2, Q_factor - b_sh);
  coeffs[3] = _float2fixed_assert( -a1 * inv_a0, Q_factor );
  coeffs[4] = _float2fixed_assert( -a2 * inv_a0, Q_factor );

  return b_sh;
}

left_shift_t adsp_design_biquad_linkwitz(
  q2_30 coeffs[5],
  const float f0,
  const float fs,
  const float q0,
  const float fp,
  const float qp
) {
  float f0_sat = _check_fc(f0, fs);
  float fp_sat = _check_fc(fp, fs);

  // Compute common factors
  float fc = (f0_sat + fp_sat) / 2.0f;

  float w_f0 = 2.0f * pi * f0_sat;
  float half_w_fc = pi * fc;
  float w_fp = 2.0f * pi * fp_sat;

  float w_f0_pow2 = w_f0 * w_f0;
  float d1i = w_f0 / q0;

  float w_fp_pow2 = w_fp * w_fp;
  float c1i = w_fp / qp;

  float gn = 2.0f * half_w_fc * (1.0f / (tanf(half_w_fc / fs)));
  float gn_pow2 = gn * gn;

  float factor_b = gn * d1i;
  float factor_a = gn * c1i;

  // Compute coeffs
  float  a0 = w_fp_pow2 + factor_a + gn_pow2;
  float  a1 = 2.0f * (w_fp_pow2 - gn_pow2);
  float  a2 = w_fp_pow2 - factor_a + gn_pow2;
  float  b0 = w_f0_pow2 + factor_b + gn_pow2;
  float  b1 = 2.0f * (w_f0_pow2 - gn_pow2);
  float  b2 = w_f0_pow2 - factor_b + gn_pow2;

  float inv_a0 = 1.0f / a0;

  b0 *= inv_a0;
  b1 *= inv_a0;
  b2 *= inv_a0;

  left_shift_t b_sh = _get_b_shift(b0, b1, b2);

  // Store as fixed-point values
  coeffs[0] = _float2fixed_assert(  b0, Q_factor - b_sh);
  coeffs[1] = _float2fixed_assert(  b1, Q_factor - b_sh);
  coeffs[2] = _float2fixed_assert(  b2, Q_factor - b_sh);
  coeffs[3] = _float2fixed_assert( -a1 * inv_a0, Q_factor );
  coeffs[4] = _float2fixed_assert( -a2 * inv_a0, Q_factor );

  return b_sh;
}

left_shift_t adsp_apply_biquad_gain(q2_30 coeffs[5], left_shift_t b_sh, float gain_db)
{
  if (gain_db == 0){
    return b_sh;
  }

  float A  = powf(10.0f, (gain_db * (1.0f / 20.0f)));
  int Q = 31 - MAX(0, (int)ceilf(log2f(A)));

  int32_t A_int = _float2fixed(A, Q);

  for(int i = 0; i < 3; i++){
    int32_t ah = 0, al = 1 << (Q - 1);
    asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (coeffs[i]), "r" (A_int), "0" (ah), "1" (al));
    asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (Q), "0" (ah), "1" (al));
    asm("lextract %0, %1, %2, %3, 32": "=r" (coeffs[i]): "r" (ah), "r" (al), "r" (Q));
  }

  return b_sh + (31 - Q);
}
