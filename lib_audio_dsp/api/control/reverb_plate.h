// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "dsp/adsp.h"
#include "helpers.h"
#include <stdint.h>

reverb_plate_t adsp_reverb_plate_init(
  float fs,
  float decay,
  float damping,
  float diffusion,
  float bandwidth,
  float in_diffusion_1,
  float in_diffusion_2,
  float width,
  float wet_gain,
  float dry_gain,
  float pregain,
  float max_predelay,
  float predelay,
  void * reverb_heap);
