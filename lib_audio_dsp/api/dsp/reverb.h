// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include <stdint.h>
#include <xmath/types.h>

#define ADSP_RV_MAX_SAMPLING_FREQ (48000.0f) // Max supported sampling freq
#define ADSP_RV_MAX_ROOM_SIZE (4.0f)         // Max supported room size

#define ADSP_RV_MIN_WET_GAIN_DB (-186.0)
#define ADSP_RV_MAX_WET_GAIN_DB (0)
#define ADSP_RV_MIN_DRY_GAIN_DB (-186.0)
#define ADSP_RV_MAX_DRY_GAIN_DB (0)

#define ADSP_RV_SCALE(FS, MAX_ROOM_SZ) (((FS) / 44100.0f) * (MAX_ROOM_SZ))

#define ADSP_RV_SUM_DEFAULT_COMB_LENS 11024
#define ADSP_RV_SUM_DEFAULT_AP_LENS 1563
#define ADSP_RV_SUM_DEFAULT_BUF_LENS (ADSP_RV_SUM_DEFAULT_COMB_LENS + \
                                      ADSP_RV_SUM_DEFAULT_AP_LENS)
#define ADSP_RV_HEAP_SZ(FS, ROOM_SZ) ((uint32_t)(sizeof(int32_t) *            \
                                                 ADSP_RV_SCALE(FS, ROOM_SZ) * \
                                                 ADSP_RV_SUM_DEFAULT_BUF_LENS))

#define ADSP_RV_N_COMBS 8
#define ADSP_RV_N_APS 4

typedef struct
{
    uint32_t max_delay;
    uint32_t delay;
    int32_t feedback;
    int32_t *buffer;
    int32_t buffer_idx;
} allpass_fv_t;

typedef struct
{
    uint32_t max_delay;
    uint32_t delay;
    int32_t feedback;
    int32_t *buffer;
    int32_t buffer_idx;
    int32_t filterstore;
    int32_t damp_1;
    int32_t damp_2;
} comb_fv_t;

typedef struct
{
    uint32_t total_buffer_length;
    uint32_t room_size;
    int32_t wet_gain;
    int32_t dry_gain;
    int32_t pre_gain;
    comb_fv_t combs[ADSP_RV_N_COMBS];
    allpass_fv_t allpasses[ADSP_RV_N_APS];
} reverb_room_t;

reverb_room_t adsp_reverb_room_init(
    float fs,
    float max_room_size,
    float room_size,
    float decay,
    float damping,
    int32_t wet_gain,
    int32_t dry_gain,
    float pregain,
    void *reverb_heap);

void adsp_reverb_room_reset_state(reverb_room_t *rv);

uint32_t adsp_reverb_room_get_buffer_lens(reverb_room_t *rv);

void adsp_reverb_room_set_room_size(
    reverb_room_t *rv,
    float new_room_size);

int32_t adsp_reverb_room(
    reverb_room_t *rv,
    int32_t new_samp);

int32_t adsp_reverb_calc_wet_gain(float wet_gain_db, float pregain);

int32_t adsp_reverb_calc_dry_gain(float dry_gain_db);
