// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "dsp/adsp.h"
#include "helpers.h"
#include <stdint.h>

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
