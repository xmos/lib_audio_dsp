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
