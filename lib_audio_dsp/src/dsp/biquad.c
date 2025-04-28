// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"
#include "stdio.h"
#include <stdbool.h>
#include <string.h>


void adsp_biquad_slew_coeffs(
  biquad_slew_t* slew_state,
  int32_t** states,
  int32_t channels
){

  if (slew_state->remaining_shifts > 0){
    // change in b_shift to manage, target_coeffs have less headroom, so add the headroom back
    for (int i=0; i < 3; i++){
      int32_t tmp_target = slew_state->target_coeffs[i] >> slew_state->remaining_shifts;
      slew_state->active_coeffs[i] += (tmp_target - slew_state->active_coeffs[i]) >> slew_state->slew_shift;
    }  
    for (int i=3; i < 5; i++){
      slew_state->active_coeffs[i] += (slew_state->target_coeffs[i] - slew_state->active_coeffs[i]) >> slew_state->slew_shift;
    }

    // see if we have headroom to do the shift
    int32_t max_val = (1 << 30);
    bool res = true;
    for (int i=0; i < 3; i++){
      res = res && (slew_state->active_coeffs[i] < max_val);
      res = res && (slew_state->active_coeffs[i] > -max_val);
    }
    for (int i=0; i < channels; i++){
      res = res && (states[i][3] < max_val);
      res = res && (states[i][3] > -max_val);
      res = res && (states[i][4] < max_val);
      res = res && (states[i][4] > -max_val);
    }

    if (res){
      // we now have the headroom to shift
      for (int i=0; i < 3; i++){
        slew_state->active_coeffs[i] <<= 1;
      }
      for (int i=0; i < channels; i++){
        states[i][3] <<= 1;
        states[i][4] <<= 1;
      }
      slew_state->remaining_shifts -= 1;
      slew_state->lsh -= 1;
    }
  }
  else{
    for (int i=0; i < 5; i++){
      slew_state->active_coeffs[i] += (slew_state->target_coeffs[i] - slew_state->active_coeffs[i]) >> slew_state->slew_shift;
    }
  }

}
