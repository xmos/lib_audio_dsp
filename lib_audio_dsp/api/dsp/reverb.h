// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include <stdint.h>
#include <xmath/types.h>

#define MAX_CHANS 1

#define DEFAULT_COMB_LENS                              \
    {                                                  \
        1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617 \
    }
#define DEFAULT_AP_LENS    \
    {                      \
        556, 441, 341, 225 \
    }
#define SUM_DEFAULT_COMB_LENS 11024
#define SUM_DEFAULT_AP_LENS 1593
#define SUM_DEFAULT_BUF_LENS (SUM_DEFAULT_COMB_LENS + SUM_DEFAULT_AP_LENS)

#define RV_SCALE(FS, MAX_ROOM_SZ) (((FS) / 44100.0f) * (MAX_ROOM_SZ))
#define RV_HEAP_SZ(FS, MAX_ROOM_SZ) ((uint32_t)(sizeof(int32_t) *           \
                                                RV_SCALE(FS, MAX_ROOM_SZ) * \
                                                SUM_DEFAULT_BUF_LENS))

#define N_COMBS 8
#define N_APS 4
#define MAX_CHANS 1
#define Q_RV 31
#define DEFAULT_PREGAIN 0.015f

#define MIN_WET_GAIN_DB (-186.0)
#define MAX_WET_GAIN_DB (0)

typedef struct
{
    void *heap_start;
    uint32_t num_bytes;
    uint32_t allocated_bytes;
} mem_manager_t;

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
    uint32_t room_size;
    int32_t wet_gain;
    int32_t dry_gain;
    int32_t pre_gain;
    uint32_t comb_lengths[N_COMBS];
    uint32_t ap_lengths[N_APS];
    comb_fv_t combs[N_COMBS];
    allpass_fv_t allpasses[N_APS];
} reverb_room_chan_t;

typedef struct
{
    reverb_room_chan_t channel[MAX_CHANS];
    uint32_t n_chans;
    uint32_t buffer_length;
} reverb_room_t;

reverb_room_t adsp_reverb_room_init(
    uint32_t n_chans,
    float fs,
    float max_room_size,
    float room_size,
    float decay,
    float damping,
    int32_t wet_gain,
    int32_t dry_gain,
    float pregain,
    void *reverb_heap);

void adsp_reverb_room_reset_state(reverb_room_t *reverb_unit);

uint32_t adsp_reverb_room_get_buffer_lens(reverb_room_t *reverb_unit);

void adsp_reverb_room_set_room_size(
    reverb_room_t *reverb_unit,
    float new_room_size);

int32_t adsp_reverb_room(
    reverb_room_t *reverb_unit,
    int32_t new_samp,
    uint32_t channel);

// Definitely don't expose these once released
allpass_fv_t allpass_fv_init(
    uint32_t max_delay,
    uint32_t starting_delay,
    int32_t feedback_gain,
    mem_manager_t *mem);

int32_t allpass_fv(allpass_fv_t *ap, int32_t new_sample);

comb_fv_t comb_fv_init(
    uint32_t max_delay,
    uint32_t starting_delay,
    int32_t feedback_gain,
    int32_t damping,
    mem_manager_t *mem);

int32_t comb_fv(comb_fv_t *comb, int32_t new_sample);

mem_manager_t mem_manager_init(
    void *heap_address,
    uint32_t number_of_bytes);

int32_t adsp_reverb_calc_wet_gain(float wet_gain_db, float pregain);
int32_t adsp_reverb_calc_dry_gain(float dry_gain_db);
