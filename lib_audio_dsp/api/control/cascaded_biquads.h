// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "xmath/types.h"

/**
 * @brief Design Nth order Butterworth lowpass filter
 * Generate N/2 sets of biquad coefficients for a Butterworth low-pass
 * filter.
 * 
 * The function implements the algorithm described in Neil Robertson's article:
 * `"Designing Cascaded Biquad Filters Using the Pole-Zero Method"
 * <https://www.dsprelated.com/showarticle/1137.php>`_.

 * It uses the bilinear transform to convert the analog filter poles to the z-plane.
 * 
 * @param coeffs    Butterworth lowpass filter coefficients
 * @param N         Order of the filter (must be even)
 * @param fc        Central frequency (-3 dB)
 * @param fs        Sampling frequency
 */
void adsp_design_butterworth_lowpass_8b(
  q2_30 coeffs[40],
  const unsigned N,
  const float fc,
  const float fs);

/**
 * @brief Design Nth order Butterworth highpass filter
 * Generate N/2 sets of biquad coefficients for a Butterworth high-pass
 * filter.
 * 
 * The function implements the algorithm described in Neil Robertson's article:
 * `"Designing Cascaded Biquad Filters Using the Pole-Zero Method"
 * <https://www.dsprelated.com/showarticle/1137.php>`_.

 * It uses the bilinear transform to convert the analog filter poles to the z-plane.
 * 
 * @param coeffs    Butterworth highpass filter coefficients
 * @param N         Order of the filter (must be even)
 * @param fc        Central frequency (-3 dB)
 * @param fs        Sampling frequency
 */
void adsp_design_butterworth_highpass_8b(
  q2_30 coeffs[40],
  const unsigned N,
  const float fc,
  const float fs);
