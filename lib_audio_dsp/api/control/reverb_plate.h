// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "dsp/adsp.h"
#include "helpers.h"
#include <stdint.h>

#if Q_RVP != Q_RVR
#error "Some reverb plate APIs are an alias from the reverb room ones, so exponents need to be the same"
#endif

/**
 * @brief Convert a floating point value to the Q_RVP format, saturate out of
 * range values. Accepted range is 0 to 1
 *
 * @param x A floating point number, will be capped to [0, 1]
 * @return Q_RVP int32_t value
 */
static inline int32_t adsp_reverb_plate_float2int(float x) {
  return _positive_float2fixed_saturate(x, Q_RVP);
}

/**
 * @brief Calculate the reverb gain in linear scale
 * 
 * Will convert a gain in dB to a linear scale in Q_RVP format.
 * To be used for converting wet and dry gains for the plate reverb.
 * 
 * @param gain_db           Gain in dB 
 * @return int32_t          Linear gain in a Q_RVP format
 */
static inline int32_t adsp_reverb_plate_calc_gain(float gain_db) {
  return db_to_qxx(gain_db, Q_RVP);
}

/**
 * @brief Calculate the stereo wet gains of the reverb plate
 * 
 * @param wet_gains       Output linear wet_1 and wet_2 gains in Q_RVP
 * @param wet_gain        Input wet gain in dB
 * @param width           Stereo separation [0, 1]
 */
static inline void adsp_reverb_plate_calc_wet_gains(int32_t wet_gains[2], float wet_gain, float width) {
  return adsp_reverb_room_st_calc_wet_gains(wet_gains, wet_gain, width);
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
