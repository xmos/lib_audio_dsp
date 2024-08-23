// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "dsp/reverb.h"


/**
 * @brief Calculate the reverb gain in linear scale
 * 
 * Will convert a gain in dB to a linear scale in Q_RVR format.
 * To be used for converting wet and dry gains for the room_reverb.
 * 
 * @param gain_db           Gain in dB
 * @return int32_t          Linear gain in a Q_RVR format
 */
int32_t adsp_reverb_room_calc_gain(float gain_db);

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
