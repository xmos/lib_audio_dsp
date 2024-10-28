// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include <xmath/types.h>

/** Reverb plate scale factor for the sampling fequency */
#define ADSP_RVP_SCALE(FS) ((float)FS / 29761)

/** Default reverb plate buffer length */
#define ADSP_RVP_SUM_DEFAULT_BUF_LENS (22467)
/** Heap size to allocate for the reverb plate */
#define ADSP_RVP_HEAP_SZ(FS, PD) ((uint32_t)((sizeof(int32_t) * \
                                          ADSP_RVP_SCALE(FS) * \
                                          ADSP_RVP_SUM_DEFAULT_BUF_LENS) + \
                                          DELAY_DSP_REQUIRED_MEMORY_SAMPLES(PD)))
/** External API for calculating memory to allocate for the reverb plate */
#define REVERB_PLATE_DSP_REQUIRED_MEMORY(FS, PD) ADSP_RVP_HEAP_SZ(FS, PD)

/** Numper of paths reverb plate is splitted in */
#define ADSP_RVP_N_PATHS 2
/** Number components used to produce the signle path output in reverb plate */
#define ADSP_RVP_N_OUT_TAPS 7
/** Number of lowpass filters used in the reverb plate */
#define ADSP_RVP_N_LPS 3
/** Number of allpass filters used in the reverb plate */
#define ADSP_RVP_N_APS 6
/** Number of delay lines used in reverb plate */
#define ADSP_RVP_N_DELAYS 4
/** Reverb plate internal Q factor */
#define Q_RVP 31

/** 
 * @brief A generic first order lowpass filter.
 */
typedef struct {
  /** State variables for lowpass filter */
  int32_t filterstore;
  /** Damping coefficient 1 */
  int32_t damp_1;
  /** Damping coefficient 2 */
  int32_t damp_2;
} lowpass_1ord_t;

/**
 * @brief A plate reverb structure
 */
typedef struct {
  /** Reverb decay */
  int32_t decay;
  /** Wet 1 linear gain */
  int32_t wet_gain1;
  /** Wet 2 linear gain */
  int32_t wet_gain2;
  /** Dry linear gain */
  int32_t dry_gain;
  /** Linear pre-gain */
  int32_t pre_gain;
  /** Saved output paths*/
  int32_t paths[ADSP_RVP_N_PATHS];
  /** Indexes for the left channel calculation */
  int32_t taps_l[ADSP_RVP_N_OUT_TAPS];
  /** Max lenghts of buffers use for the left channel calculation */
  int32_t taps_len_l[ADSP_RVP_N_OUT_TAPS];
  /** Indexes for the right channel calculation */
  int32_t taps_r[ADSP_RVP_N_OUT_TAPS];
  /** Max lenghts of buffers use for the right channel calculation */
  int32_t taps_len_r[ADSP_RVP_N_OUT_TAPS];
  /** FIrst order lowpass filters */
  lowpass_1ord_t lowpasses[ADSP_RVP_N_LPS];
  /** Modulated allpass filters */
  allpass_fv_t mod_allpasses[ADSP_RVP_N_PATHS];
  /** Allpass filters */
  allpass_fv_t allpasses[ADSP_RVP_N_APS];
  /** Delay lines */
  delay_t delays[ADSP_RVP_N_DELAYS];
  /** Predelay applied to the wet channel*/
  delay_t predelay;
} reverb_plate_t;

/**
 * @brief Initialise first order lowpass
 *
 * @param feedback        Feedback coefficient
 * @return lowpass_1ord_t Initialised lowpass object
 */
lowpass_1ord_t lowpass_1ord_init(int32_t feedback);

/**
 * @brief Lower level function to initialise the filters of a reverb plate object
 * 
 * Will initialise allpasses, modulated allpasses, delays and predelay.
 * 
 * @param rv                Reverb plate object
 * @param fs                Sampling frequency
 * @param decay_diffusion_1 Late diffusion
 * @param decay_diffusion_2 Diffusion
 * @param in_diffusion_1    Early diffusion 1
 * @param in_diffusion_2    Early diffusion 2
 * @param max_predelay      Maximum size of the predelay buffer in samples
 * @param predelay          Initial predelay in samples
 * @param reverb_heap       Pointer to heap to allocate reverb memory
 */
void adsp_reverb_plate_init_filters(
  reverb_plate_t * rv,
  float fs,
  int32_t decay_diffusion_1,
  int32_t decay_diffusion_2,
  int32_t in_diffusion_1,
  int32_t in_diffusion_2,
  uint32_t max_predelay,
  uint32_t predelay,
  void * reverb_heap);

/**
 * @brief Process samples through a reverb plate object
 * 
 * @param rv                Reverb plate object
 * @param outputs_lr        Pointer to the outputs 0:left, 1:right
 * @param in_left           New left sample to process
 * @param in_right          New right sample to process
 */
void adsp_reverb_plate(
  reverb_plate_t *rv,
  int32_t outputs_lr[2],
  int32_t in_left,
  int32_t in_right);
