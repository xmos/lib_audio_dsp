// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once
#include <xcore/assert.h>    // for xassert()

#define Q_alpha (31)

/**
 * @brief Saturating rounding multiply by a Q0.31 gain.
 * 
 * @param samp Sample to be multipled.
 * @param gain Gain to apply; assumes a Q0.31 gain.
 * @return int32_t Returns either samp * gain or MAXINT/MININT if over/underflow
 */
static inline int32_t apply_gain_q31(int32_t samp, q1_31 gain) {
  // this assumes that alpha is q31
  int32_t q = Q_alpha;
  int32_t ah = 0, al = 1 << (q - 1);
  // standard multiplication with rounding and saturation
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (samp), "r" (gain), "0" (ah), "1" (al));
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (q), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (q));
  return ah;
}


static inline int32_t _float2fixed( float x, int32_t q )
{
  if     ( x < 0 ) return (((float)(1 << q))       * x - 0.5);
  else if( x > 0 ) return (((float)((1 << q) - 1)) * x + 0.5);
  return 0;
}


/**
 * @brief Convert a value in decibels to a fixed point int32 number in
 *        the given Q format.
 * 
 * @param level_db Level in db
 * @param q_format Required Q format of the output
 * @return int32_t level_db as an int32_t
 */
static inline int32_t db_to_q_format(float level_db, int q_format) {
  float A  = powf(10, (level_db / 20));
  int32_t out = _float2fixed( A, q_format);
  return out;
}


/**
 * @brief Convert a power level in decibels to a fixed point int32 number in
 *        the given Q format.
 * 
 * @param level_db Power level in db
 * @param q_format Required Q format of the output
 * @return int32_t level_db as an int32_t
 */
static inline int32_t db_pow_to_q_format(float level_db, int q_format) {
  float A  = powf(10, (level_db / 10));
  int32_t out = _float2fixed( A, q_format);
  return out;
}


/**
 * @brief Convert a fixed point int32 number in the given Q format to a
 *        value in decibels.
 * 
 * @param level Level in the fixed point format specified by q_format
 * @param q_format Q format of the input
 * @return float level in dB for the signal
 */
static inline float q_format_to_db(int32_t level, int q_format) {
  float level_db = 20.0*log10f((float)level / (float)(1 << q_format));
  return level_db;
}

/**
 * @brief Convert a fixed point int32 number in the given Q format to a
 *        value in decibels, when the input level is power.
 * 
 * @param level Power level in the fixed point format specified by q_format
 * @param q_format Q format of the input
 * @return float level in dB for the signal
 */
static inline float q_format_to_db_pow(int32_t level, int q_format) {
  float level_db = 10.0*log10f((float)level / (float)(1 << q_format));
  return level_db;
}


/**
 * @brief Convert an attack or release time in seconds to an EWM alpha 
 *        value as a fixed point int32 number in Q_alpha format.
 * 
 * @param fs sampling frequency
 * @param time attack/release time in seconds
 * @return int32_t attack/release alpha as an int32_t
 */
static inline int32_t calc_alpha(float fs, float time) {
  xassert(time > 0 && "time has to be positive");
  time = 2 / (fs * time);
  int32_t sign, exp, mant;
  asm("fsexp %0, %1, %2": "=r" (sign), "=r" (exp): "r" (time));
  asm("fmant %0, %1": "=r" (mant): "r" (time));

  // mant to q31
  right_shift_t shr = -Q_alpha - exp + 23;
  mant >>= shr;
  return mant;
}

static inline int32_t calc_peak_threshold(float level_db){
  db_to_q_format(level_db, Q_SIG);
}

static inline int32_t calc_rms_threshold(float level_db){
  db_pow_to_q_format(level_db, Q_SIG);
}

/**
 * @brief Convert a compressor ratio to the slope, where the slope is
 *        defined as (1 - 1 / ratio) / 2.0. The division by 2 compensates for
 *        the RMS envelope detector returning the RMS².
 */
static inline float rms_compressor_slope_from_ratio(float ratio){
  float slope = (1.0 - 1.0 / ratio) / 2.0;
  return slope;

}

/**
 * @brief Convert an expander ratio to the slope, where the slope is
          defined as (1 - ratio).
 */
static inline float peak_expander_slope_from_ratio(float ratio){
  float slope = 1.0 - ratio;
  return slope;
}
