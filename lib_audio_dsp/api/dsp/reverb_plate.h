
#pragma once

#include <xmath/types.h>

#define ADSP_RVP_SCALE(FS) ((float)FS / 29761)
#define ADSP_RVP_SUM_DEFAULT_BUF_LENS (22467)

/** Heap size to allocate for the reverb room */
#define ADSP_RVP_HEAP_SZ(FS, PD) ((uint32_t)((sizeof(int32_t) * \
                                          ADSP_RVP_SCALE(FS) * \
                                          ADSP_RVP_SUM_DEFAULT_BUF_LENS) + \
                                          DELAY_DSP_REQUIRED_MEMORY_SAMPLES(PD)))

#define REVERB_PLATE_DSP_REQUIRED_MEMORY(FS, PD) ADSP_RVP_HEAP_SZ(FS, PD)

#define ADSP_RVP_N_OUT_TAPS 7
#define ADSP_RVP_N_LPS 3
#define ADSP_RVP_N_PATHS 2
#define ADSP_RVP_N_APS 6
#define ADSP_RVP_N_DELAYS 4
#define Q_RVP 31

typedef struct {
  int32_t filterstore;
  int32_t damp_1;
  int32_t damp_2;
} lowpass_1ord_t;

typedef struct {
  int32_t decay;
  int32_t wet_gain1;
  int32_t wet_gain2;
  int32_t dry_gain;
  int32_t pre_gain;
  int32_t paths[ADSP_RVP_N_PATHS];
  int32_t taps_l[ADSP_RVP_N_OUT_TAPS];
  int32_t taps_len_l[ADSP_RVP_N_OUT_TAPS];
  int32_t taps_r[ADSP_RVP_N_OUT_TAPS];
  int32_t taps_len_r[ADSP_RVP_N_OUT_TAPS];
  lowpass_1ord_t lowpasses[ADSP_RVP_N_LPS];
  allpass_fv_t mod_allpasses[ADSP_RVP_N_PATHS];
  allpass_fv_t allpasses[ADSP_RVP_N_APS];
  delay_t delays[ADSP_RVP_N_DELAYS];
  delay_t predelay;
} reverb_plate_t;

lowpass_1ord_t lowpass_1ord_init(int32_t feedback);

void adsp_reverb_plate_init_filters(
  reverb_plate_t * rv,
  float fs,
  int32_t decay,
  int32_t diffusion,
  int32_t in_diffusion_1,
  int32_t in_diffusion_2,
  uint32_t max_predelay, 
  uint32_t predelay, 
  void * reverb_heap);

void adsp_reverb_plate(
  reverb_plate_t *rv,
  int32_t outputs_lr[2],
  int32_t in_left,
  int32_t in_right);
