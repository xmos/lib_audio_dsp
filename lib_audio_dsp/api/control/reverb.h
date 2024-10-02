// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "dsp/adsp.h"
#include "helpers.h"
#include <stdint.h>

/// Convert a floating point value to the Q_VERB format, saturate out of
/// range values. Accepted range is 0 to 1
///
/// @param x A floating point number
/// @return postive Q_VERB int32_t value
static inline int32_t adsp_reverb_float2int(float x) {
    return _float2fixed_saturate(x < 0.0f ? 0.0f : x, Q_RVR);
}

/// Convert a floating point gain in decibels into a linear Q_VERB value
/// for use in controlling the reverb gains. 
/// 
/// @param db Floating point value in dB, values above 0 will be clipped.
/// @return Q_VERB fixed point linear gain.
static inline int32_t adsp_reverb_db2int(float db) {
    float a  = powf(10.0f, (db / 20.0f));
    int32_t out = adsp_reverb_float2int(a);
    return out;
}

/// Convert a user damping value into a Q_VERB fixed point value suitable
/// for passing to a reverb.
///
/// @param damping The chose value of damping.
/// @return Damping as a Q_VERB fixed point integer, clipped to the accepted range.
static inline int32_t adsp_reverb_calculate_damping(float damping) {
    int32_t ret = adsp_reverb_float2int(damping);
    return ret < 1 ? 1 : ret;
}

/// Calculate a Q_VERB feedback value for a given decay. Use to calculate 
/// the feedback parameter in reverb_room.
///
/// @param decay The desired decay value.
/// @return Calculated feedback as a Q_VERB fixed point integer.
static inline int32_t adsp_reverb_calculate_feedback(float decay) {
    if(decay < 0.0f) {
        decay = 0.0f;
    }
    if(decay > 1.0f) {
        decay = 1.0f;
    }
    float feedback = (0.28f * decay) + 0.7f;
    return adsp_reverb_float2int(feedback);
}

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
 * @brief Calculate the wet and dry gains according to the mix amount.
 * 
 * When the mix is set to 0, only the dry signal will be output. 
 * The wet gain will be 0 and the dry gain will be max.
 * When the mic is set to 1, only they wet signal will be output. 
 * The wet gain is max, the dry gain will be 0.
 * In order to maintain a consistent signal level across all mix values, 
 * the signals are panned with a -4.5 dB panning law.
 * 
 * @param gains           Output gains: [0] - Dry; [1] - Wet
 * @param mix             Mix applied from 0 to 1
 */
void adsp_reverb_wet_dry_mix(int32_t gains[2], float mix);

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
 * @param max_predelay    Maximum size of the predelay buffer in ms
 * @param predelay        Initial predelay in ms
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
    float max_predelay,
    float predelay,
    void *reverb_heap);

/**
 * @brief Calculate the stereo wet gains on the stereo reverb room
 * 
 * @param wet_gains       Output linear wet_1 and wet_2 gains in Q_RVR
 * @param wet_gain        Input wet gain in dB
 * @param width           Stereo separation of the room [0, 1]
 */
void adsp_reverb_room_st_calc_wet_gains(int32_t wet_gains[2], float wet_gain, float width);

/**
 * @brief Calculate the stereo wet and dry gains according to the mix amount
 * 
 * When the mix is set to 0, only the dry signal will be output. 
 * The wet gain will be 0 and the dry gain will be max.
 * When the mic is set to 1, only they wet signal will be output. 
 * The wet gain is max, the dry gain will be 0.
 * In order to maintain a consistent signal level across all mix values, 
 * the signals are panned with a -4.5 dB panning law.
 * The width controls the mixing between the left and right wet channels 
 *
 * @param gains           Output gains: [0] - Dry; [1] - Wet_1; [2] - Wet_2
 * @param mix             Mix applied from 0 to 1
 * @param width           Stereo separation of the room [0, 1]
 */
void adsp_reverb_st_wet_dry_mix(int32_t gains[3], float mix, float width);

/**
 * @brief Initialise a stereo reverb room object
 * A room reverb effect based on Freeverb by Jezar at Dreampoint
 * 
 * @param fs                Sampling frequency
 * @param max_room_size     Maximum room size of delay filters
 * @param room_size         Room size compared to the maximum room size [0, 1]
 * @param decay             Length of the reverb tail [0, 1]
 * @param damping           High frequency attenuation
 * @param width             Stereo separation of the room [0, 1]
 * @param wet_gain          Wet gain in dB
 * @param dry_gain          Dry gain in dB
 * @param pregain           Linear pre-gain
 * @param max_predelay      Maximum size of the predelay buffer in ms
 * @param predelay          Initial predelay in ms
 * @param reverb_heap       Pointer to heap to allocate reverb memory
 * @return reverb_room_st_t Initialised stereo reverb room object
 */
reverb_room_st_t adsp_reverb_room_st_init(
  float fs,
  float max_room_size,
  float room_size,
  float decay,
  float damping,
  float width,
  float wet_gain,
  float dry_gain,
  float pregain,
  float max_predelay,
  float predelay,
  void *reverb_heap);
