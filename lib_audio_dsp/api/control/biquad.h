// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "xmath/types.h"

/**
 * @brief Design biquad filter bypass
 * This function creeates a bypass biquad filter. Only the b0 coefficient is set.
 * 
 * @param coeffs          Bypass filter coefficients
 */
void adsp_design_biquad_bypass(q2_30 coeffs[5]);

/**
 * @brief Design mute biquad filter
 * This function creates a mute biquad filter. All the coefficients are 0.
 * 
 * @param coeffs          Mute filter coefficients
 */
void adsp_design_biquad_mute(q2_30 coeffs[5]);

/**
 * @brief Design gain biquad filter
 * This function creates a biquad filter with a specified gain
 * 
 * @param coeffs          Gain filter coefficients
 * @param gain_db         Gain in dB
 * @return left_shift_t   Left shift compensation value
 */
left_shift_t adsp_design_biquad_gain(q2_30 coeffs[5], const float gain_db);


/**
 * @brief Design lowpass biquad filter
 * This function creates a biquad filter with a lowpass response
 * ``fc`` must be less than ``fs/2``, otherwise it will be saturated to
 * ``fs/2``.
 * 
 * @param coeffs          Lowpass filter coefficients
 * @param fc              Cutoff frequency
 * @param fs              Sampling frequency
 * @param filter_Q        Filter Q
 */
void adsp_design_biquad_lowpass(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q);

/**
 * @brief Design highpass biquad filter
 * This function creates a biquad filter with a highpass response
 * ``fc`` must be less than ``fs/2``, otherwise it will be saturated to
 * ``fs/2``.
 * 
 * @param coeffs          Highpass filter coefficients
 * @param fc              Cutoff frequency
 * @param fs              Sampling frequency
 * @param filter_Q        Filter Q
 */
void adsp_design_biquad_highpass(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q);

/**
 * @brief Design bandpass biquad filter
 * This function creates a biquad filter with a bandpass response
 *  ``fc`` must be less than ``fs/2``, otherwise it will be saturated to
 * ``fs/2``.
 * 
 * @param coeffs          Bandpass filter coefficients
 * @param fc              Central frequency
 * @param fs              Sampling frequency
 * @param bandwidth       Bandwidth
 */
void adsp_design_biquad_bandpass(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float bandwidth);

/**
 * @brief Design bandstop biquad filter
 * This function creates a biquad filter with a bandstop response
 *  ``fc`` must be less than ``fs/2``, otherwise it will be saturated to
 * ``fs/2``.
 * 
 * @param coeffs          Bandstop filter coefficients
 * @param fc              Central frequency
 * @param fs              Sampling frequency
 * @param bandwidth       Bandwidth
 */
void adsp_design_biquad_bandstop(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float bandwidth);

/**
 * @brief Design notch biquad filter
 * This function creates a biquad filter with an notch response
 *  ``fc`` must be less than ``fs/2``, otherwise it will be saturated to
 * ``fs/2``.
 * 
 * @param coeffs          Notch filter coefficients
 * @param fc              Central frequency
 * @param fs              Sampling frequency
 * @param filter_Q        Filter Q
 */
void adsp_design_biquad_notch(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q);

/**
 * @brief Design allpass biquad filter
 * This function creates a biquad filter with an allpass response
 *  ``fc`` must be less than ``fs/2``, otherwise it will be saturated to
 * ``fs/2``.
 * 
 * @param coeffs          Allpass filter coefficients
 * @param fc              Central frequency
 * @param fs              Sampling frequency
 * @param filter_Q        Filter Q
 */
void adsp_design_biquad_allpass(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q);

/**
 * @brief Design peaking biquad filter
 * This function creates a biquad filter with a peaking response
 *  ``fc`` must be less than ``fs/2``, otherwise it will be saturated to
 * ``fs/2``.
 * 
 * The gain must be less than 18 dB, otherwise the coefficients may overflow.
 * If the gain is greater than 18 dB, it is saturated to that value.
 * 
 * @param coeffs          Peaking filter coefficients
 * @param fc              Central frequency
 * @param fs              Sampling frequency
 * @param filter_Q        Filter Q
 * @param gain_db         Gain in dB
 * @return left_shift_t   Left shift compensation value
 */
left_shift_t adsp_design_biquad_peaking(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db);

/**
 * @brief Design constant Q peaking biquad filter
 * This function creates a biquad filter with a constant Q peaking response.
 * 
 * Constant Q means that the bandwidth of the filter remains constant
 * as the gain varies. It is commonly used for graphic equalisers.
 *  ``fc`` must be less than ``fs/2``, otherwise it will be saturated to
 * ``fs/2``.
 * 
 * The gain must be less than 18 dB, otherwise the coefficients may overflow.
 * If the gain is greater than 18 dB, it is saturated to that value.
 * 
 * @param coeffs          Constant Q filter coefficients
 * @param fc              Central frequency
 * @param fs              Sampling frequency
 * @param filter_Q        Filter Q
 * @param gain_db         Gain in dB
 * @return left_shift_t   Left shift compensation value
 */
left_shift_t adsp_design_biquad_const_q(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db);

/**
 * @brief Design lowshelf biquad filter
 * This function creates a biquad filter with a lowshelf response.
 * 
 * The Q factor is defined in a similar way to standard low pass, i.e.
 * Q > 0.707 will yield peakiness (where the shelf response does not
 * monotonically change). The level change at f will be boost_db/2.
 *  ``fc`` must be less than ``fs/2``, otherwise it will be saturated to
 * ``fs/2``.
 * 
 * The gain must be less than 12 dB, otherwise the coefficients may overflow.
 * If the gain is greater than 12 dB, it is saturated to that value.
 * 
 * @param coeffs          Lowshelf filter coefficients
 * @param fc              Cutoff frequency
 * @param fs              Sampling frequency
 * @param filter_Q        Filter Q
 * @param gain_db         Gain in dB
 * @return left_shift_t   Left shift compensation value
 */
left_shift_t adsp_design_biquad_lowshelf(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db);

/**
 * @brief Design highshelf biquad filter
 * This function creates a biquad filter with a highshelf response.
 * 
 * The Q factor is defined in a similar way to standard high pass, i.e.
 * Q > 0.707 will yield peakiness. The level change at f will be
 * boost_db/2. ``fc`` must be less than ``fs/2``, otherwise it will be saturated to
 * ``fs/2``.
 * 
 * The gain must be less than 12 dB, otherwise the coefficients may overflow.
 * If the gain is greater than 12 dB, it is saturated to that value.
 * 
 * @param coeffs          Highshelf filter coefficients
 * @param fc              Cutoff frequency
 * @param fs              Sampling frequency
 * @param filter_Q        Filter Q
 * @param gain_db         Gain in dB
 * @return left_shift_t   Left shift compensation value
 */
left_shift_t adsp_design_biquad_highshelf(
  q2_30 coeffs[5],
  const float fc,
  const float fs,
  const float filter_Q,
  const float gain_db);

/**
 * @brief Design Linkwitz transform biquad filter
 * This function creates a biquad filter with a Linkwitz transform response.
 * 
 * The Linkwitz Transform is commonly used to change the low frequency
 * roll off slope of a loudspeaker. When applied to a loudspeaker, it
 * will change the cutoff frequency from f0 to fp, and the quality
 * factor from q0 to qp. ``f0`` and ``fp`` must be less than ``fs/2``,
 * otherwise they will be saturated to ``fs/2``.
 * 
 * @param coeffs          Linkwitz filter coefficients
 * @param f0              Original cutoff frequency
 * @param fs              Sampling frequency
 * @param q0              Original quality factor at f0
 * @param fp              Target cutoff frequency
 * @param qp              Target quality factor of the filter
 */
void adsp_design_biquad_linkwitz(
  q2_30 coeffs[5],
  const float f0,
  const float fs,
  const float q0,
  const float fp,
  const float qp);  

