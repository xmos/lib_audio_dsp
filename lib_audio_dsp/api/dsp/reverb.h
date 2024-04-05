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
        556, 441, 341, 255 \
    }
#define SUM_DEFAULT_COMB_LENS 11024
#define SUM_DEFAULT_AP_LENS 1593
#define SUM_DEFAULT_BUF_LENS (SUM_DEFAULT_COMB_LENS + SUM_DEFAULT_AP_LENS)

#define RV_SCALE(FS, MAX_ROOM_SZ) (((FS) / 44100.0f) * (MAX_ROOM_SZ))
#define RV_HEAP_SZ(FS, MAX_ROOM_SZ) ((uint32_t)(sizeof(int32_t) *           \
                                                RV_SCALE(FS, MAX_ROOM_SZ) * \
                                                SUM_DEFAULT_BUF_LENS))

typedef struct
{
    void * channel[MAX_CHANS];
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
    float wet_gain_db,
    float dry_gain_db,
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