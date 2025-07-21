// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include <stdbool.h>

/** Heap size to allocate for the delay from samples */
#define DELAY_DSP_REQUIRED_MEMORY_SAMPLES(SAMPLES) (sizeof(int32_t) * (SAMPLES))
/** Heap size to allocate for the delay from milliseconds */
#define DELAY_DSP_REQUIRED_MEMORY_MS(FS, MS) (sizeof(int32_t) * ((FS) * (MS) / 1000))
/** Heap size to allocate for the delay from seconds */
#define DELAY_DSP_REQUIRED_MEMORY_SEC(FS, SEC) (sizeof(int32_t) * (FS) * (SEC))

/** Gain format to be used in the gain APIs */
#define Q_GAIN 27


/**
 * @brief Slewing gain state structure
 */
typedef struct{
  /** Target linear gain */
  int32_t target_gain;
  /** Current linear gain */
  int32_t gain;
  /** Slew shift */
  int32_t slew_shift;
} gain_slew_t;

/**
 * @brief Volume control state structure
 */
typedef struct{
  /** Target linear gain */
  int32_t target_gain;
  /** Current linear gain */
  int32_t gain;
  /** Slew shift */
  int32_t slew_shift;
  /** Saved linear gain */
  int32_t saved_gain;
  /** Mute state: 0: unmuted, 1 muted */
  uint8_t mute_state;
}volume_control_t;

/**
 * @brief Delay state structure
 */
typedef struct{
  /** Sampling frequency */
  float fs;
  /** Current delay in samples */
  uint32_t delay;
  /** Maximum delay in samples */
  uint32_t max_delay;
  /** Current buffer index */
  uint32_t buffer_idx;
  /** Buffer */
  int32_t * buffer;
} delay_t;


/**
 * @brief Slewing switch state structure
 */
typedef struct{
  /** If slewing, switching is True until slewing is over. */
  bool switching;
  /** Current switch pole position. */
  int32_t position;
  /** Last switch pole position. */
  int32_t last_position;
  /** Counter for timing slew length. */
  int32_t counter;
  /** Step increment of counter. */
  int32_t step;
} switch_slew_t;

/**
 * @brief Slewing crossfader state structure
 */
typedef struct{
  /** Slewing gain struct for first crossfader input. */
  gain_slew_t gain_1;
  /** Slewing gain struct for second crossfader input. */
  gain_slew_t gain_2;
  /** Mix of the inputs. */
  float mix;
} crossfader_slew_t;

/**
 * @brief Convert from Q0.31 to Q_SIG
 *
 * @param input             Input in Q0.31 format
 * @return int32_t          Output in Q_SIG format
 */
int32_t adsp_from_q31(int32_t input);

/**
 * @brief Convert from Q_SIG to Q0.31
 *
 * @param input             Input in Q_SIG format
 * @return int32_t          Output in Q0.31 format
 */
int32_t adsp_to_q31(int32_t input);

/**
 * @brief Saturating addition of an array of samples
 *
 * @param input             Array of samples
 * @param n_ch              Number of channels
 * @return int32_t          Sum of samples
 * @note Will work for any q format
 */
int32_t adsp_adder(int32_t * input, unsigned n_ch);

/**
 * @brief Saturating subtraction of two samples, this returns `x - y`.
 *
 * @param x                 Minuend
 * @param y                 Subtrahend
 * @return int32_t          Difference
 * @note Will work for any q format
 */
int32_t adsp_subtractor(int32_t x, int32_t y);

/**
 * @brief Fixed-point gain
 *
 * @param input             Input sample
 * @param gain              Gain
 * @return int32_t          Output sample
 * @note One of the inputs has to be in Q_GAIN format
 */
int32_t adsp_fixed_gain(int32_t input, int32_t gain);

/**
 * @brief Mixer.
 * Will add signals with gain applied to each signal before mixing
 *
 * @param input             Array of samples
 * @param n_ch              Number of channels
 * @param gain              Gain
 * @return int32_t          Mixed sample
 * @note Inputs or gain have to be in Q_GAIN format
 */
int32_t adsp_mixer(int32_t * input, unsigned n_ch, int32_t gain);

/**
 * @brief Saturating 64-bit accumulator.
 * Will saturate to 32-bit, so that the output value is in the range of int32_t
 *
 * @param acc               Accumulator
 * @return int32_t          Saturated value
 */
int32_t adsp_saturate_32b(int64_t acc);

/**
 * @brief Process a new sample with a volume control
 *
 * @param vol_ctl           Volume control object
 * @param samp              New sample
 * @return int32_t          Processed sample
 */
int32_t adsp_volume_control(
  volume_control_t * vol_ctl,
  int32_t samp);

/**
 * @brief Set the target gain of a volume control
 *
 * @param vol_ctl           Volume control object
 * @param new_gain          New target linear gain
 */
void adsp_volume_control_set_gain(
  volume_control_t * vol_ctl,
  int32_t new_gain);

/**
 * @brief Mute a volume control.
 * Will save the current target gain and set the target gain to 0
 *
 * @param vol_ctl           Volume control object
 */
void adsp_volume_control_mute(
  volume_control_t * vol_ctl);

/**
 * @brief Unmute a volume control.
 * Will restore the saved target gain
 *
 * @param vol_ctl           Volume control object
 */
void adsp_volume_control_unmute(
  volume_control_t * vol_ctl);

/**
 * @brief Process a new sample through a delay object
 * 
 * @note The minimum delay provided by this block is 1 sample. Setting
 *       the delay to 0 will still yield a 1 sample delay.
 *
 * @param delay             Delay object
 * @param samp              New sample
 * @return int32_t          Oldest sample
 */
int32_t adsp_delay(
  delay_t * delay,
  int32_t samp);

/**
 * @brief Process a sample through a slewing switch. If the switch
 * position has recently changed, this will slew between the desired
 * input channel and previous channel.
 * 
 * @param switch_slew    Slewing switch state object.
 * @param samples        An array of input samples for each input channel.
 * @return int32_t       The output of the switch.
 */
int32_t adsp_switch_slew(switch_slew_t* switch_slew, int32_t* samples);

/**
 * @brief Exponentially slew the gain towards the target gain.
 * 
 * @param gain_state    Slewing gain state object.
 * @return int32_t      The current gain.
 */
static inline int32_t adsp_slew_gain(gain_slew_t * gain_state) {
  // do the exponential slew
  gain_state->gain += (gain_state->target_gain - gain_state->gain) >> gain_state->slew_shift;
  return gain_state->gain;
}

/**
 * @brief Crossfade between two channels using their gains.
 * Will do: (in1 * gain1) + (in2 * gain2).
 *
 * @param in1      First signal
 * @param in2      Second signal
 * @param gain1     First gain
 * @param gain2     Second gain
 * @param q_gain    Q factor of the gain
 * @return int32_t  Mixed signal
 */
static inline int32_t adsp_crossfader(int32_t in1, int32_t in2, int32_t gain1, int32_t gain2, int32_t q_gain) {
  int32_t ah = 0, al = 1 << (q_gain - 1);
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (in1), "r" (gain1), "0" (ah), "1" (al));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (in2), "r" (gain2), "0" (ah), "1" (al));
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (q_gain), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (q_gain));
  return ah;
}

/**
 * @brief Crossfade between two channels with slew applied to the gains.
 * Will do: (in1 * crossfader->gain1.gain) + (in2 * crossfader->gain2.gain).
 *
 * @param crossfader    Slewing crossfader state object.
 * @param in1           First signal
 * @param in2           Second signal
 * @return int32_t      Mixed signal
 */
int32_t adsp_crossfader_slew(crossfader_slew_t* crossfader, int32_t in1, int32_t in2);

/**
 * @brief Process samples through the 4:1 router
 *
 * Takes an array of 4 input samples and routes active channels to a single output
 * according to the channel_states array in the router_4to1_t structure.
 * Active channels (set to true) are summed together.
 *
 * @param router            Pointer to a 4 long array of channel states (0 = disabled, 1 = enabled)
 * @param samples           Array of 4 input samples
 * @return int32_t          Mixed output sample
 */
int32_t adsp_router_4to1(int32_t* router_states, int32_t* samples);
