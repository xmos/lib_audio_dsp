// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include <xmath/types.h>

/** Minimum wet/dry gain config for the reverb room in dB */
#define ADSP_RVR_MIN_GAIN_DB (-186.0)
/** Maximum wet/dry gain config for the reverb room in dB */
#define ADSP_RVR_MAX_GAIN_DB (0)

/** Reverb room scale factor for the room size */
#define ADSP_RVR_SCALE(FS, MAX_ROOM_SZ) (((FS) / 44100.0f) * (MAX_ROOM_SZ))

/** Default reverb room buffer length */
#define ADSP_RVR_SUM_DEFAULT_BUF_LENS 12587
/** Heap size to allocate for the reverb room */
#define ADSP_RVR_HEAP_SZ(FS, ROOM_SZ) ((uint32_t)(sizeof(int32_t) *            \
                                                 ADSP_RVR_SCALE(FS, ROOM_SZ) * \
                                                 ADSP_RVR_SUM_DEFAULT_BUF_LENS))
/** External API for calculating memory to allocate for the reverb room */
#define REVERB_ROOM_DSP_REQUIRED_MEMORY(FS, ROOM_SZ) ADSP_RVR_HEAP_SZ(FS, ROOM_SZ)

/** Number of comb filters used in the reverb room */
#define ADSP_RVR_N_COMBS 8
/** Number of allpass filters used in the reverb room */
#define ADSP_RVR_N_APS 4
/** Reverb room internal Q factor */
#define Q_RVR 31

/**
 * @brief A freeverb style all-pass filter structure
 */
typedef struct
{
    /** Maximum delay */
    uint32_t max_delay;
    /** Current delay */
    uint32_t delay;
    /** Feedback gain */
    int32_t feedback;
    /** Delay buffer */
    int32_t *buffer;
    /** Current buffer index */
    int32_t buffer_idx;
} allpass_fv_t;

/**
 * @brief A freeverb style comb filter structure
 */
typedef struct
{
    /** Maximum delay */
    uint32_t max_delay;
    /** Current delay */
    uint32_t delay;
    /** Feedback gain */
    int32_t feedback;
    /** Delay buffer */
    int32_t *buffer;
    /** Current buffer index */
    int32_t buffer_idx;
    /** State variables for low-pass filter */
    int32_t filterstore;
    /** Damping coefficient 1 */
    int32_t damp_1;
    /** Damping coefficient 2 */
    int32_t damp_2;
} comb_fv_t;

/**
 * @brief A room reverb filter structure
 */
typedef struct
{
    /** Total buffer length */
    uint32_t total_buffer_length;
    /** Room size */
    float room_size;
    /** Wet linear gain */
    int32_t wet_gain;
    /** Dry linear gain */
    int32_t dry_gain;
    /** Linear pre-gain */
    int32_t pre_gain;
    /** Comb filters */
    comb_fv_t combs[ADSP_RVR_N_COMBS];
    /** Allpass filters */
    allpass_fv_t allpasses[ADSP_RVR_N_APS];
} reverb_room_t;

/**
 * @brief Lower level function to initialise the filters of a reverb room object
 * 
 * Will only initialise allpass, comb filters and set total buffer length.
 * Can be used before `adsp_room_reverb_set_room_size()` to
 * initialise the filters and set the rooms size.
 * 
 * feedback can be calculated from the decay parameter as follows:
 * `feedback = Q_RVR((decay * 0.28f) + 0.7f)`
 * 
 * @param rv                Reverb room object
 * @param fs                Sampling frequency
 * @param max_room_size     Maximum room size of delay filters
 * @param feedback          Feedback gain for the comb filters in Q_RVR format
 * @param damping           Damping coefficient for the comb filters in Q_RVR format
 * @param reverb_heap       Pointer to heap to allocate reverb memory
 */
void adsp_reverb_room_init_filters(
    reverb_room_t *rv,
    float fs,
    float max_room_size,
    int32_t feedback,
    int32_t damping,
    void * reverb_heap);

/**
 * @brief Reset the state of a reverb room object
 * 
 * @param rv                Reverb room object
 */
void adsp_reverb_room_reset_state(reverb_room_t *rv);

/**
 * @brief Get the buffer length of a reverb room object
 * 
 * @param rv                Reverb room object
 * @return uint32_t         Buffer length
 */
uint32_t adsp_reverb_room_get_buffer_lens(reverb_room_t *rv);

/**
 * @brief Set the room size of a reverb room object
 * 
 * @param rv                Reverb room object
 * @param new_room_size     New room size [0, 1]
 */
void adsp_reverb_room_set_room_size(
    reverb_room_t *rv,
    float new_room_size);

/**
 * @brief Process a sample through a reverb room object
 * 
 * @param rv                Reverb room object
 * @param new_samp          New sample to process
 * @return int32_t          Processed sample
 */
int32_t adsp_reverb_room(
    reverb_room_t *rv,
    int32_t new_samp);

