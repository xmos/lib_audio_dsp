
#include "dsp/adsp.h"

#include <xcore/assert.h>

#define Q_factor 30
#define BOOST_BSHIFT 2

static const float pi =    3.14159265359;
static const float log_2 = 0.69314718055;

static inline int32_t _float2fixed( float x, int32_t q )
{
  if     ( x < 0 ) return (((float)(1 << q))       * x - 0.5);
  else if( x > 0 ) return (((float)((1 << q) - 1)) * x + 0.5);
  return 0;
}

void adsp_design_biquad_bypass(int32_t coeffs[5]) {
  coeffs[0] = 1 << Q_factor;
	coeffs[1] = 0;
	coeffs[2] = 0;
	coeffs[3] = 0;
	coeffs[4] = 0;
}

void adsp_design_biquad_mute(int32_t coeffs[5]) {
  coeffs[0] = 0;
	coeffs[1] = 0;
	coeffs[2] = 0;
	coeffs[3] = 0;
	coeffs[4] = 0;
}

void adsp_design_biquad_lowpass
(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q
) {
  xassert(fc <= fs / 2 && "fc must be less than fs/2");
  // Compute common factors
	float w0 = 2.0 * pi * fc / fs;
	float alpha = f32_sin(w0) / (2.0 * filter_Q);

  // Compute coeffs
	float b0 = (1.0 - f32_cos(w0)) / 2.0;
	float b1 = (1.0 - f32_cos(w0));
	float b2 =  b0;
	float a0 =  1.0 + alpha;
	float a1 = -2.0 * f32_cos(w0);
	float a2 =  1.0 - alpha;
	
	// Store as fixed-point values
	coeffs[0] = _float2fixed(  b0 / a0, Q_factor );
	coeffs[1] = _float2fixed(  b1 / a0, Q_factor );
	coeffs[2] = _float2fixed(  b2 / a0, Q_factor );
	coeffs[3] = _float2fixed( -a1 / a0, Q_factor );
	coeffs[4] = _float2fixed( -a2 / a0, Q_factor );
}

void adsp_design_biquad_highpass
(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q
) {
  xassert(fc <= fs / 2 && "fc must be less than fs/2");
  // Compute common factors
	float w0    = 2.0 * pi * fc / fs;
	float alpha = f32_sin(w0) / (2.0 * filter_Q);

  // Compute coeffs
	float b0 =  (1.0 + f32_cos(w0)) / 2.0;
	float b1 = -(1.0 + f32_cos(w0));
	float b2 =   b0;
	float a0 =   1.0 + alpha;
	float a1 =  -2.0 * f32_cos(w0);
	float a2 =   1.0 - alpha;
	
	// Store as fixed-point values
	coeffs[0] = _float2fixed(  b0 / a0, Q_factor );
	coeffs[1] = _float2fixed(  b1 / a0, Q_factor );
	coeffs[2] = _float2fixed(  b2 / a0, Q_factor );
	coeffs[3] = _float2fixed( -a1 / a0, Q_factor );
	coeffs[4] = _float2fixed( -a2 / a0, Q_factor );
}


void adsp_design_biquad_bandpass
(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float bandwidth
) {
  xassert(fc <= fs / 2 && "fc must be less than fs/2");
  // Compute common factors
	float w0    = 2.0 * pi * fc / fs;
  float alpha = f32_sin(w0) * sinhf(log_2 / 2 * bandwidth * w0 / f32_sin(w0));

  // Compute coeffs
  float b0 =  alpha;
  float b1 =  0.0;
  float b2 = -alpha;
  float a0 =  1.0 + alpha;
  float a1 = -2.0 * f32_cos(w0);
  float a2 =  1.0 - alpha;
	
  // Store as fixed-point values
	coeffs[0] = _float2fixed(  b0 / a0, Q_factor );
	coeffs[1] = _float2fixed(  b1 / a0, Q_factor );
	coeffs[2] = _float2fixed(  b2 / a0, Q_factor );
	coeffs[3] = _float2fixed( -a1 / a0, Q_factor );
	coeffs[4] = _float2fixed( -a2 / a0, Q_factor );
}

void adsp_design_biquad_bandstop
(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float bandwidth
) {
  xassert(fc <= fs / 2 && "fc must be less than fs/2");
  // Compute common factors
	float w0    = 2.0 * pi * fc / fs;
  float alpha = f32_sin(w0) * sinhf(log_2 / 2 * bandwidth * w0 / f32_sin(w0));
	
  // Compute coeffs
  float b0 =  1.0;
  float b1 = -2.0 * f32_cos(w0);
  float b2 =  1.0;
  float a0 =  1.0 + alpha;
  float a1 =  b1;
  float a2 =  1.0 - alpha;

  // Store as fixed-point values
	coeffs[0] = _float2fixed(  b0 / a0, Q_factor );
	coeffs[1] = _float2fixed(  b1 / a0, Q_factor );
	coeffs[2] = _float2fixed(  b2 / a0, Q_factor );
	coeffs[3] = _float2fixed( -a1 / a0, Q_factor );
	coeffs[4] = _float2fixed( -a2 / a0, Q_factor );
}

void adsp_design_biquad_notch
(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q
) {
  xassert(fc <= fs / 2 && "fc must be less than fs/2");
  // Compute common factors
	float w0    = 2.0 * pi * fc / fs;
	float alpha = f32_sin(w0) / (2.0 * filter_Q);

  // Compute coeffs
	float b0 =  1.0;
	float b1 = -2.0 * f32_cos(w0);
	float b2 =  1.0;
	float a0 =  1.0 + alpha;
	float a1 =  b1;
	float a2 =  1.0 - alpha;
	
	// Store as fixed-point values
	coeffs[0] = _float2fixed(  b0 / a0, Q_factor );
	coeffs[1] = _float2fixed(  b1 / a0, Q_factor );
	coeffs[2] = _float2fixed(  b2 / a0, Q_factor );
	coeffs[3] = _float2fixed( -a1 / a0, Q_factor );
	coeffs[4] = _float2fixed( -a2 / a0, Q_factor );
}

void adsp_design_biquad_allpass
(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q
) {
  xassert(fc <= fs / 2 && "fc must be less than fs/2");
  // Compute common factors
	float w0    = 2.0 * pi * fc / fs;
	float alpha = f32_sin(w0) / (2.0 * filter_Q);

  // Compute coeffs
	float b0 =  1.0 - alpha;
	float b1 = -2.0 * f32_cos(w0);
	float b2 =  1.0 + alpha;
	float a0 =  1.0 + alpha;
	float a1 =  b1;
	float a2 =  1.0 - alpha;
	
	// Store as fixed-point values
	coeffs[0] = _float2fixed(  b0 / a0, Q_factor );
	coeffs[1] = _float2fixed(  b1 / a0, Q_factor );
	coeffs[2] = _float2fixed(  b2 / a0, Q_factor );
	coeffs[3] = _float2fixed( -a1 / a0, Q_factor );
	coeffs[4] = _float2fixed( -a2 / a0, Q_factor );
}

left_shift_t adsp_design_biquad_peaking
(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db
) {
  xassert(fc <= fs / 2 && "fc must be less than fs/2");
  // Compute common factors
	float A  = sqrtf(powf(10, (gain_db / 20)));
	float w0 = 2.0 * pi * fc / fs;
	float alpha = f32_sin(w0) / (2.0 * filter_Q);

  // Compute coeffs
	float b0 =  1.0 + alpha * A;
	float b1 = -2.0 * f32_cos(w0);
	float b2 =  1.0 - alpha * A;
	float a0 =  1.0 + alpha / A;
	float a1 =  b1;
	float a2 =  1.0 - alpha / A;
	
	// Store as fixed-point values
	coeffs[0] = _float2fixed(  b0 / a0, Q_factor - BOOST_BSHIFT );
	coeffs[1] = _float2fixed(  b1 / a0, Q_factor - BOOST_BSHIFT );
	coeffs[2] = _float2fixed(  b2 / a0, Q_factor - BOOST_BSHIFT );
	coeffs[3] = _float2fixed( -a1 / a0, Q_factor );
	coeffs[4] = _float2fixed( -a2 / a0, Q_factor );

  return BOOST_BSHIFT;
}

left_shift_t adsp_design_biquad_const_q
(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db
) {
  xassert(fc <= fs / 2 && "fc must be less than fs/2");
  // Compute common factors
  float V = powf(10, (gain_db / 20));
  // w0 is only needed for calculating K
  float K = tanf(pi * fc / fs);

  float factor_a = K / filter_Q;
  float factor_b = 0;
  float K_pow2 = K * K;
  if(gain_db > 0) {
    factor_b = V * factor_a;
  }
  else {
    factor_b = factor_a;
    factor_a = factor_b / V;
  }

  // Compute coeffs
  float b0 = 1 + factor_b + K_pow2;
  float b1 = 2 * (K_pow2  - 1);
  float b2 = 1 - factor_b + K_pow2;
  float a0 = 1 + factor_a + K_pow2;
  float a1 = b1;
  float a2 = 1 - factor_a + K_pow2;
  
	// Store as fixed-point values
	coeffs[0] = _float2fixed(  b0 / a0, Q_factor - BOOST_BSHIFT );
	coeffs[1] = _float2fixed(  b1 / a0, Q_factor - BOOST_BSHIFT );
	coeffs[2] = _float2fixed(  b2 / a0, Q_factor - BOOST_BSHIFT );
	coeffs[3] = _float2fixed( -a1 / a0, Q_factor );
	coeffs[4] = _float2fixed( -a2 / a0, Q_factor );

  return BOOST_BSHIFT;
}

left_shift_t adsp_design_biquad_lowshelf
(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db
) {
  xassert(fc <= fs / 2 && "fc must be less than fs/2");
  // Compute common factors
	float A  = powf(10, (gain_db / 40));
	float w0 = 2.0 * pi * fc / fs;
	float alpha = f32_sin(w0) / (2.0 * filter_Q);

  float alpha_factor = 2 * sqrtf(A) * alpha;
  float Am1_cosw0 = (A - 1) * f32_cos(w0);
  float Ap1_cosw0 = (A + 1) * f32_cos(w0);
  
  // Compute coeffs
  float b0 =  A * ((A + 1) - Am1_cosw0 + alpha_factor);
  float b1 =  2 * A * ((A - 1) - Ap1_cosw0);
  float b2 =  A * ((A + 1) - Am1_cosw0 - alpha_factor);
  float a0 = (A + 1) + Am1_cosw0 + alpha_factor;
  float a1 = -2 * ((A - 1) + Ap1_cosw0);
  float a2 = (A + 1) + Am1_cosw0 - alpha_factor;
	
  // Store as fixed-point values
	coeffs[0] = _float2fixed(  b0 / a0, Q_factor - BOOST_BSHIFT );
	coeffs[1] = _float2fixed(  b1 / a0, Q_factor - BOOST_BSHIFT );
	coeffs[2] = _float2fixed(  b2 / a0, Q_factor - BOOST_BSHIFT );
	coeffs[3] = _float2fixed( -a1 / a0, Q_factor );
	coeffs[4] = _float2fixed( -a2 / a0, Q_factor );

  return BOOST_BSHIFT;
}

left_shift_t adsp_design_biquad_highshelf
(
  int32_t coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db
) {
  xassert(fc <= fs / 2 && "fc must be less than fs/2");
  // Compute common factors
	float A  = powf(10, (gain_db / 40));
	float w0 = 2.0 * pi * fc / fs;
	float alpha = f32_sin(w0) / (2.0 * filter_Q);

  float alpha_factor = 2 * sqrtf(A) * alpha;
  float Am1_cosw0 = (A - 1) * f32_cos(w0);
  float Ap1_cosw0 = (A + 1) * f32_cos(w0);
  
  // Compute coeffs
  float b0 =  A * ((A + 1) + Am1_cosw0 + alpha_factor);
  float b1 = -2 * A * ((A - 1) + Ap1_cosw0);
  float b2 =  A * ((A + 1) + Am1_cosw0 - alpha_factor);
  float a0 = (A + 1) - Am1_cosw0 + alpha_factor;
  float a1 =  2 * ((A - 1) - Ap1_cosw0);
  float a2 = (A + 1) - Am1_cosw0 - alpha_factor;
	
  // Store as fixed-point values
	coeffs[0] = _float2fixed(  b0 / a0, Q_factor - BOOST_BSHIFT );
	coeffs[1] = _float2fixed(  b1 / a0, Q_factor - BOOST_BSHIFT );
	coeffs[2] = _float2fixed(  b2 / a0, Q_factor - BOOST_BSHIFT );
	coeffs[3] = _float2fixed( -a1 / a0, Q_factor );
	coeffs[4] = _float2fixed( -a2 / a0, Q_factor );

  return BOOST_BSHIFT;
}

void adsp_design_biquad_linkwitz(
  int32_t coeffs[5],
  const float f0,
  const float fs,
  const float q0,
  const float fp,
  const float qp
) {
  xassert(fp <= fs / 2 && "fc must be less than fs/2");
  xassert(f0 <= fs / 2 && "fc must be less than fs/2");
  // Compute common factors
  float fc = (f0 + fp) / 2;

  float d0i = 2 * pi * fc;
  float d1i = d0i / q0;
  d0i = d0i * d0i;

  float c0i = 2 * pi * fp;
  float c1i = c0i / qp;
  c0i = c0i * c0i;

  float gn = (2 * pi * fc) / (tanf(pi * fc / fs));
  float gn_pow2 = gn * gn;
  float factor_b = gn * d1i;
  float factor_a = gn * c1i;

  // Compute coeffs
  float b0 = d0i + factor_b + gn_pow2;
  float b1 = 2 * (d0i - gn_pow2);
  float b2 = d0i - factor_b + gn_pow2;
  float a0 = c0i + factor_a + gn_pow2;
  float a1 = 2 * (c0i - gn_pow2);
  float a2 = c0i - factor_a + gn_pow2;

  // Store as fixed-point values
	coeffs[0] = _float2fixed(  b0 / a0, Q_factor );
	coeffs[1] = _float2fixed(  b1 / a0, Q_factor );
	coeffs[2] = _float2fixed(  b2 / a0, Q_factor );
	coeffs[3] = _float2fixed( -a1 / a0, Q_factor );
	coeffs[4] = _float2fixed( -a2 / a0, Q_factor );
}
