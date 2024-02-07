
#include "dsp/adsp.h"

#include <math.h>
#include <xcore/assert.h>

env_detector_t adsp_env_detector_init(
  float fs,
  float attack_t,
  float release_t,
  float detect_t
) {
  env_detector_t env_det;

  if (detect_t && (attack_t || release_t)) {
    xassert(0 && "either detect_t OR (attack_t AND release_t) must be specified");
  } else if (detect_t) {
    attack_t = detect_t;
    release_t = detect_t;
  }

  env_det.attack_alpha = 2 / (fs * attack_t);
  env_det.release_alpha = 2 / (fs * release_t);
  env_det.envelope = 0;

  return env_det;
}

void adsp_env_detector_peak(
  env_detector_t * env_det,
  int32_t new_sample
) {
  float samp = float_s32_to_float((float_s32_t){new_sample, SIG_EXP});
  samp = fabsf(samp);

  float alpha = env_det->release_alpha;
  if (samp > env_det->envelope) {
    alpha = env_det->attack_alpha;
  }

  env_det->envelope = ((1 - alpha) * env_det->envelope) + (alpha * samp);
}

void adsp_env_detector_rms(
  env_detector_t * env_det,
  int32_t new_sample
) {
  float samp = float_s32_to_float((float_s32_t){new_sample, SIG_EXP});
  samp *= samp;

  float alpha = env_det->release_alpha;
  if (samp > env_det->envelope) {
    alpha = env_det->attack_alpha;
  }

  env_det->envelope = ((1 - alpha) * env_det->envelope) + (alpha * samp);
}
