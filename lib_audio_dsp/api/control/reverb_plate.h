// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "dsp/adsp.h"
#include "helpers.h"
#include <stdint.h>

#if Q_RVP != Q_RVR
#error "Some reverb room APIs are used for the reverb plate, so exponents need to be the same"
#endif

/**
 * @brief Convert a user late diffusion value into a Q_RVP fixed point value suitable
 * for passing to a reverb.
 *
 * @param late_diffusion The chose value of late diffusion.
 * @return Late diffusion as a Q_RVP fixed point integer, clipped to the accepted range.
 */
static inline int32_t adsp_reverb_plate_calc_late_diffusion(float late_diffusion) {
  return -_positive_float2fixed_saturate(late_diffusion, Q_RVP);
}

/**
 * @brief Convert a user damping value into a Q_RVP fixed point value suitable
 * for passing to a reverb.
 *
 * @param damping The chose value of damping.
 * @return Damping as a Q_RVP fixed point integer, clipped to the accepted range.
 */
static inline int32_t adsp_reverb_plate_calc_damping(float damping) {
  int32_t damp = INT32_MAX - _positive_float2fixed_saturate(damping, Q_RVP);
  return (damp < 1) ? 1 : damp;
}

/**
 * @brief Convert a user bandwidth value into a Q_RVP fixed point value suitable
 * for passing to a reverb.
 *
 * @param bandwidth The chose value of bandwidth.
 * @return Bandwidth as a Q_RVP fixed point integer, clipped to the accepted range.
 */
static inline int32_t adsp_reverb_plate_calc_bandwidth(float bandwidth) {
  int32_t band = _positive_float2fixed_saturate(bandwidth, Q_RVP);
  return (band < 1) ? 1 : band;
}

/**
 * @brief Initialise a reverb plate object
 * 
 * @param fs                Sampling frequency
 * @param decay             Length of the reverb tail [0, 1]
 * @param damping           High frequency attenuation
 * @param bandwidth         Pre lowpass
 * @param early_diffusion   Early diffusion
 * @param late_diffusion    Late diffusion
 * @param width             Stereo separation of the room [0, 1]
 * @param wet_gain          Wet gain in dB
 * @param dry_gain          Dry gain in dB
 * @param pregain           Linear pre-gain
 * @param max_predelay      Maximum size of the predelay buffer in ms
 * @param predelay          Initial predelay in ms
 * @param reverb_heap       Pointer to heap to allocate reverb memory
 * @return reverb_plate_t   Initialised reverb plate object
 */
reverb_plate_t adsp_reverb_plate_init(
  float fs,
  float decay,
  float damping,
  float bandwidth,
  float early_diffusion,
  float late_diffusion,
  float width,
  float wet_gain,
  float dry_gain,
  float pregain,
  float max_predelay,
  float predelay,
  void * reverb_heap);
