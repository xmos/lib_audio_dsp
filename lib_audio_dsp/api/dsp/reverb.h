// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include <stdint.h>
#include <xmath/types.h>

/** Minimum wet/dry gain config for the reverb in dB */
#define ADSP_RV_MIN_GAIN_DB (-186.0)
/** Maximum wet/dry gain config for the reverb in dB */
#define ADSP_RV_MAX_GAIN_DB (0)

/** Reverb scale factor for the room size */
#define ADSP_RV_SCALE(FS, MAX_ROOM_SZ) (((FS) / 44100.0f) * (MAX_ROOM_SZ))

/** Default reverb buffer length */
#define ADSP_RV_SUM_DEFAULT_BUF_LENS 12587
/** Heap size to allocate for the reverb */
#define ADSP_RV_HEAP_SZ(FS, ROOM_SZ) ((uint32_t)(sizeof(int32_t) *            \
                                                 ADSP_RV_SCALE(FS, ROOM_SZ) * \
                                                 ADSP_RV_SUM_DEFAULT_BUF_LENS))
/** External API for calculating memory to allocate for the reverb*/
#define REVERB_DSP_REQUIRED_MEMORY(FS, ROOM_SZ) ADSP_RV_HEAP_SZ(FS, ROOM_SZ)

/** Number of comb filters used in the reverb */
#define ADSP_RV_N_COMBS 8
/** Number of allpass filters used in the reverb */
#define ADSP_RV_N_APS 4

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
    uint32_t room_size;
    /** Wet linear gain */
    int32_t wet_gain;
    /** Dry linear gain */
    int32_t dry_gain;
    /** Linear pre-gain */
    int32_t pre_gain;
    /** Comb filters */
    comb_fv_t combs[ADSP_RV_N_COMBS];
    /** Allpass filters */
    allpass_fv_t allpasses[ADSP_RV_N_APS];
} reverb_room_t;

/**
 * @brief Initialise a reverb room object
 * A room reverb effect based on Freeverb by Jezar at Dreampoint
 * 
 * @param fs              Sampling frequency
 * @param max_room_size   Maximum room size of delay filters
 * @param room_size       Room size compared to the maximum room size [0, 1]
 * @param decay           Lenght of the reverb tail [0, 1]
 * @param damping         High frequency attenuation
 * @param wet_gain        Wet gain in dB
 * @param dry_gain        Dry gain in dB
 * @param pregain         Linear pre-gain
 * @param reverb_heap     Pointer to heap to allocate reverb memory
 * @return reverb_room_t  Initialised reverb room object
 */
reverb_room_t adsp_reverb_room_init(
    float fs,
    float max_room_size,
    float room_size,
    float decay,
    float damping,
    float wet_gain,
    float dry_gain,
    float pregain,
    void *reverb_heap);

/**
 * @brief Lower level function to initialise the filters of a reverb room object
 * 
 * Will only initilise allpass, comb filters and set total buffer length.
 * Can be used before `adsp_room_reverb_set_room_size()` to
 * initialise the filters and set the rooms size.
 * 
 * feedback can be calculated from the decay parameter as follows:
 * `feedback = Q_RV((decay * 0.28f) + 0.7f)`
 * 
 * @param rv                Reverb room object
 * @param fs                Sampling frequency
 * @param max_room_size     Maximum room size of delay filters
 * @param feedback          Feedback gain for the comb filters in Q_RV format
 * @param damping           Damping coefficient for the comb filters in Q_RV format
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

/**
 * @brief Calculate the reverb gain in linear scale
 * 
 * Will convert a gain in dB to a linear scale in Q_RV format.
 * To be used for converting wet and dry gains for the room_reverb.
 * 
 * @param gain_db           Gain in dB
 * @return int32_t          Linear gain in a Q_RV format
 */
int32_t adsp_reverb_calc_gain(float gain_db);
