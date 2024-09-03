// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once
#include <xcore/assert.h>    // for xassert()
#include <math.h>
#include <limits.h>
#include "xmath/xmath.h"
#include <dsp/_helpers/generic_utils.h> // for Q_alpha
#include <dsp/defines.h>

static const float pi =    (float)M_PI;
static const float log_2 = 0.69314718055f;
static const float db_2 = 6.02059991328f;  // 20*log10(2)


/**
 * @brief Convert a float value to a fixed point int32 number in
 *        q format. If the value of x is outside the fixed point range,
 *        this will overflow.
 * 
 * @param x A floating point value
 * @param q Q format of the output
 * @return int32_t x in q fixed point format
 */
static inline int32_t _float2fixed( float x, int32_t q )
{
  if     ( x < 0 ) return (((float)(1 << q))       * x - 0.5f);
  else if( x > 0 ) return (((float)((1 << q) - 1)) * x + 0.5f);
  return 0;
}


/**
 * @brief Convert a float value to a fixed point int32 number in
 *        q format. If the value of x is outside the fixed point range,
 *        this will raise an assertion.
 * 
 * @param x A floating point value
 * @param q Q format of the output
 * @return int32_t x in q fixed point format
 */
static inline int32_t _float2fixed_assert( float x, int32_t q )
{
  float max_val = (float)(1<<(31-q));
  xassert(x < max_val); // Too much gain, cannot be represented in desired number format
  xassert(x >= -max_val);

  if     ( x < 0 ) return (((float)(1 << q))       * x - 0.5f);
  else if( x > 0 ) return (((float)((1 << q) - 1)) * x + 0.5f);
  return 0;
}


/**
 * @brief Convert a float value to a fixed point int32 number in
 *        q format. If the value of x is outside the fixed point range,
 *        it is saturated.
 * 
 * @param x A floating point value
 * @param q Q format of the output
 * @return int32_t x in q fixed point format
 */
static inline int32_t _float2fixed_saturate( float x, int32_t q )
{
  if (x < -(1 << (31-q))) return INT32_MIN;
  else if ( x < 0 ) return (((float)(1 << q)) * x - 0.5f);
  else if ( x >= 1 << (31-q)) return INT32_MAX;
  else if( x > 0 ) return (((float)((1 << q) - 1)) * x + 0.5f);
  return 0;
}

/**
 * @brief Convert a positive float value to a fixed point int32 number in
 *        Q_SIG format. By assuming the value is positive (e.g. a gain value
 *        converted from decibels), negative cases can be ignored. If the
 *        value of x exceeds the fixed point maximum, it is saturated.
 * 
 * @param x A positive floating point value
 * @return int32_t x in Q_SIG fixed point format
 */
static inline int32_t _positive_float2fixed_qsig(float x)
{
  if ( x >= 1 << (31-Q_SIG)) return INT32_MAX;
  else if( x > 0 ) return (((float)((1 << Q_SIG) - 1)) * x + 0.5f);
  return 0;
}


/**
 * @brief Convert a value in decibels to a fixed point int32 number in
 *        Q_SIG format. If the level exceeds the fixed point maximum,
 *        it is saturated.
 * 
 * @param level_db Level in db
 * @return int32_t level_db as an int32_t
 */
static inline int32_t db_to_q_sig(float level_db) {
  float A  = powf(10.0f, (level_db / 20.0f));
  int32_t out = _positive_float2fixed_qsig(A);
  return out;
}


/**
 * @brief Convert a power level in decibels to a fixed point int32 number in
 *        Q_SIG format. If the level exceeds the fixed point maximum,
 *        it is saturated.
 * 
 * @param level_db Power level in db
 * @return int32_t level_db in Q_SIG fixed point format
 */
static inline int32_t db_pow_to_q_sig(float level_db) {
  float A  = powf(10.0f, (level_db / 10.0f));
  int32_t out = _positive_float2fixed_qsig(A);
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
static inline float qxx_to_db(int32_t level, int q_format) {
  float level_db = 20.0f*log10f((float)level / (float)(1 << q_format));
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
static inline float qxx_to_db_pow(int32_t level, int q_format) {
  float level_db = 10.0f*log10f((float)level / (float)(1 << q_format));
  return level_db;
}


/**
 * @brief Convert an attack or release time in seconds to an EWM alpha 
 *        value as a fixed point int32 number in Q_alpha format. If the
 *        desired time is too large or small to be represented in the fixed
 *        point format, it is saturated.
 * 
 * @param fs sampling frequency in Hz
 * @param time attack/release time in seconds
 * @return int32_t attack/release alpha as an int32_t
 */
static inline int32_t calc_alpha(float fs, float time) {
  float alpha = 1.0f;
  if (time > 0.0f){
    alpha = 2.0f / (fs * time);
    alpha = MIN(alpha, 1.0f);
  }

  int32_t mant;

  if(alpha == 1.0f){
    mant = INT32_MAX;
  }
  else{
  #ifdef __XS3A__
    // multiply alpha by 2**31 and convert to an int32
    int32_t sign, exp;
    asm("fsexp %0, %1, %2": "=r" (sign), "=r" (exp): "r" (alpha));
    asm("fmant %0, %1": "=r" (mant): "r" (alpha));
    // mant to q31
    right_shift_t shr = -Q_alpha - exp + 23;
    mant >>= shr;
  #else
    mant = (int32_t)(alpha * (float)(1u << 31));
  #endif
  }

  return mant;
}


/**
 * @brief Convert a peak compressor/limiter/expander threshold in decibels
 *        to an int32 fixed point gain in Q_SIG Q format.
 *        If the threshold is higher than representable in the fixed point
 *        format, it is saturated.
 *        The minimum threshold returned by this function is 1.
 *
 * @param level_db the desired threshold in decibels
 * @return int32_t the threshold as a fixed point integer.
 */
static inline int32_t calculate_peak_threshold(float level_db){
  int32_t out = db_to_q_sig(level_db);
  out = MAX(out, 1);
  return out;
}


/**
 * @brief Convert an RMS² compressor/limiter/expander threshold in decibels
 *        to an int32 fixed point gain in Q_SIG Q format.
 *        If the threshold is higher than representable in the fixed point
 *        format, it is saturated.
 *        The minimum threshold returned by this function is 1.
 *
 * @param level_db the desired threshold in decibels
 * @return int32_t the threshold as a fixed point integer.
 */
static inline int32_t calculate_rms_threshold(float level_db){
  int32_t out = db_pow_to_q_sig(level_db);
  out = MAX(out, 1);
  return out;
}


/**
 * @brief Convert a compressor ratio to the slope, where the slope is
 *        defined as (1 - 1 / ratio) / 2.0. The division by 2 compensates for
 *        the RMS envelope detector returning the RMS². The ratio must be
 *        greater than 1, if it is not the ratio is set to 1.
 *
 * @param ratio the desired compressor ratio
 * @return float slope of the compressor
 */
static inline float rms_compressor_slope_from_ratio(float ratio){
  ratio = MAX(ratio, 1.0f);
  float slope = (1.0f - 1.0f / ratio) / 2.0f;
  return slope;

}


/**
 * @brief Convert an expander ratio to the slope, where the slope is
 *        defined as (1 - ratio). The ratio must be
 *        greater than 1, if it is not the ratio is set to 1.
 *
 * @param ratio the desired expander ratio
 * @return float slope of the expander
 */
static inline float peak_expander_slope_from_ratio(float ratio){
  ratio = MAX(ratio, 1.0f);
  float slope = 1.0f - ratio;
  return slope;
}
