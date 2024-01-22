
#include "dsp/cascaded_biquads.h"
#include "dsp/biquad.h"
#include <xcore/assert.h>
#include <stdio.h>

#define Q_factor 30
static const float pi =    3.14159265359;

static inline int32_t _float2fixed( float x, int32_t q )
{
  if     ( x < 0 ) return (((float)(1 << q))       * x - 0.5);
  else if( x > 0 ) return (((float)((1 << q) - 1)) * x + 0.5);
  return 0;
}

static inline void _get_pa(complex_float_t * pa, unsigned N) {
  unsigned n_filts = N / 2;
  for (unsigned k = 0; k < n_filts; k++) {
    float theta = (2 * (k + 1) - 1) * pi / (2 * N);
    pa[n_filts - 1 - k].re = -f32_sin(theta);
    pa[n_filts - 1 - k].im = f32_cos(theta);
  }
}

// returns (1 + val) / (1 - val)
static inline complex_float_t _get_p(complex_float_t val) {
  complex_float_t one_plus = val;
  complex_float_t one_minus = val;

  one_plus.re = 1 + one_plus.re;
  one_minus.re = 1 - one_minus.re;
  one_minus.im = - one_minus.im;

  float denom = one_minus.re * one_minus.re + one_minus.im * one_minus.im;
  float num = one_plus.re * one_minus.re + one_plus.im * one_minus.im;
  val.re = num / denom;
  num  = one_plus.im * one_minus.re - one_plus.re * one_minus.im;
  val.im = num / denom;
  return val;
}

int32_t adsp_cascaded_biquads_8b(int32_t new_sample,
                                int32_t coeffs[40],
                                int32_t state[64],
                                left_shift_t lsh[8]
) {
  int32_t out = new_sample;
  for(unsigned n = 0; n < 8; n++)
  {
    out = adsp_biquad(out, &coeffs[5 * n], &state[8 * n], lsh[n]);
  }
  return out;
}

void adsp_design_butterworth_lowpass_8b(
  int32_t coefficients[40],
  const unsigned N,
  const float fc,
  const float fs
) {
  xassert(N % 2 == 0 && "N must be even");
  xassert(fc <= fs / 2 && "fc must be less than fs/2");
  complex_float_t pa[8] = {{0}};

  float factor = pi / fs;
  float Fc = fs / pi * tanf(factor * fc);
  factor *= Fc;

  _get_pa(pa, N);

  for (unsigned i = 0; i < N / 2; i ++) {
    complex_float_t val = pa[i];
    val.re *= factor;
    val.im *= factor;

    val = _get_p(val);

    float a1 = -2 * val.re;
    float a2 = val.re * val.re + val.im * val.im;

    float b0 = (1 + a1 + a2) / 4;
    float b1 = 2 * b0;
    float b2 = b0;

    coefficients[5 * i]     = _float2fixed(  b0 , Q_factor );
    coefficients[5 * i + 1] = _float2fixed(  b1 , Q_factor );
    coefficients[5 * i + 2] = _float2fixed(  b2 , Q_factor );
    coefficients[5 * i + 3] = _float2fixed( -a1 , Q_factor );
    coefficients[5 * i + 4] = _float2fixed( -a2 , Q_factor );
  }
  for (unsigned i = N / 2; i < 8; i ++) {
    adsp_design_biquad_bypass(&coefficients[5 * i]);
  }
}

void adsp_design_butterworth_highpass_8b(
  int32_t coefficients[40],
  const unsigned N,
  const float fc,
  const float fs
) {
  xassert(N % 2 == 0 && "N must be even");
  xassert(fc <= fs / 2 && "fc must be less than fs/2");
  complex_float_t pa[8] = {{0}};

  float factor = pi / fs;
  float Fc = fs / pi * tanf(factor * fc);
  factor *= Fc;

  _get_pa(pa, N);
  for (unsigned i = 0; i < N / 2; i ++) {
    float den = pa[i].re * pa[i].re + pa[i].im * pa[i].im;
    float num = factor * pa[i].re;
    pa[i].re = num / den;
    num = - factor * pa[i].im;
    pa[i].im = num / den;

    pa[i] = _get_p(pa[i]);

    float a1 = -2 * pa[i].re;
    float a2 = pa[i].re * pa[i].re + pa[i].im * pa[i].im;

    float b0 = (1 - a1 + a2) / 4;
    float b1 = - 2 * b0;
    float b2 = b0;

    coefficients[5 * i]     = _float2fixed(  b0 , Q_factor );
    coefficients[5 * i + 1] = _float2fixed(  b1 , Q_factor );
    coefficients[5 * i + 2] = _float2fixed(  b2 , Q_factor );
    coefficients[5 * i + 3] = _float2fixed( -a1 , Q_factor );
    coefficients[5 * i + 4] = _float2fixed( -a2 , Q_factor );
  }
  for (unsigned i = N / 2; i < 8; i ++) {
    adsp_design_biquad_bypass(&coefficients[5 * i]);
  }
}