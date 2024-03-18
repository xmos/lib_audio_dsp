// Copyright 2024 XMOS ngITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"

static inline int32_t apply_gain(int32_t samp, float gain) {
  int32_t mant, exp, zero;
  asm("fsexp %0, %1, %2": "=r" (zero), "=r" (exp): "r" (gain));
  asm("fmant %0, %1": "=r" (mant): "r" (gain));

  int32_t q = -(exp - 23);
  int32_t ah = 0, al = 1 << (q - 1);
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (samp), "r" (mant), "0" (ah), "1" (al));
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (q), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (q));

  return ah;
}

noise_gate_t adsp_noise_gate_init(
  float fs,
  float threshold_db,
  float atack_t,
  float release_t
) {
  return (noise_gate_t)adsp_limiter_peak_init(fs, threshold_db, atack_t, release_t);
}

int32_t adsp_noise_gate(
  noise_gate_t * ng,
  int32_t new_samp
) {
  adsp_env_detector_peak(&ng->env_det, new_samp);
  float env = (ng->env_det.envelope == 0) ? 1e-20 : ng->env_det.envelope;
  float new_gain = (ng->threshold > env) ? 0 : 1;

  // for the noise gate, the attack and release times are swapped
  // i.e. attack time is after going under threshold instead of over
  float alpha = ng->env_det.attack_alpha;
  if( ng->gain > new_gain ) {
    alpha = ng->env_det.release_alpha;
  }

  ng->gain = ng->gain + alpha * (new_gain - ng->gain);
  return apply_gain(new_samp, ng->gain);
}
