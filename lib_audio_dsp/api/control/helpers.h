// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once
#include <xcore/assert.h>    // for xassert()
#include <math.h>
#include <limits.h>
#include "xmath/xmath.h"
#include <dsp/_helpers/generic_utils.h> // for Q_alpha
#include <dsp/defines.h>

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
#ifdef __XS3A__
  int32_t sign, exp, mant;
  asm("fsexp %0, %1, %2": "=r" (sign), "=r" (exp): "r" (x));
  asm("fmant %0, %1": "=r" (mant): "r" (x));
  if(sign){mant = -mant;}
  // mant to q
  right_shift_t shr = -q - exp + 23;
  return mant >>= shr;
#else
  if ( x < 0.0f ) return (((float)(1u << q))       * x - 0.5f);
  if ( x > 0.0f ) return (((float)((1u << q) - 1)) * x + 0.5f);
  return 0;
#endif
}

/**
 * @brief Convert a float value to a fixed point int32 number in
 *        q format. If the value of x is outside the positive 
 *        fixed point range,this will overflow.
 * 
 * @param x A floating point value
 * @param q Q format of the output
 * @return int32_t x in q fixed point format
 */
static inline int32_t _positive_float2fixed(float x, int32_t q)
{
#ifdef __XS3A__
  int32_t sign, exp, mant;
  asm("fsexp %0, %1, %2": "=r" (sign), "=r" (exp): "r" (x));
  asm("fmant %0, %1": "=r" (mant): "r" (x));
  // mant to q
  right_shift_t shr = -q - exp + 23;
  return mant >>= shr;
#else
  return ((float)((1u << q) - 1)) * x + 0.5f
#endif
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
  xassert(x <= max_val); // Too much gain, cannot be represented in desired number format
  xassert(x > -max_val);

  return _float2fixed(x, q);
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
  if (x >= (1 << (31-q))) return INT32_MAX;

  return _float2fixed(x, q);
}

/**
 * @brief Convert a float value to a fixed point int32 number in
 *        q format. Negative input will result in the output of zero.
 *        If the value of x is outside the fixed point range,
 *        it is saturated.
 * 
 * @param x A floating point value
 * @param q Q format of the output
 * @return int32_t x in q fixed point format
 */
static inline int32_t _positive_float2fixed_saturate(float x, int32_t q)
{
  if ( x <= 0.0f ) return 0;
  if ( x >= (1 << (31-q))) return INT32_MAX;

  return _positive_float2fixed(x, q);
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
  return _positive_float2fixed_saturate(x, Q_SIG);
}

/**
 * @brief Convert a value in decibels to a fixed point int32 number in
 *        a given q format. If the level exceeds the fixed point maximum,
 *        it is saturated.
 * 
 * @param level_db Level in db
 * @param q Q format of the output
 * @return int32_t level_db as an int32_t
 */
static inline int32_t db_to_qxx(float level_db, int32_t q) {
  float A  = powf(10.0f, (level_db / 20.0f));
  int32_t out = _positive_float2fixed_saturate(A, q);
  return out;
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
  return db_to_qxx(level_db, Q_SIG);
}

/**
 * @brief Convert a power level in decibels to a fixed point int32 number in
 *        a given q format. If the level exceeds the fixed point maximum,
 *        it is saturated.
 * 
 * @param level_db Power level in db
 * @param q Q format of the output
 * @return int32_t level_db as an int32_t
 */
static inline int32_t db_pow_to_qxx(float level_db, int32_t q) {
  float A  = powf(10.0f, (level_db / 10.0f));
  int32_t out = _positive_float2fixed_saturate(A, q);
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
  return db_pow_to_qxx(level_db, Q_SIG);
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
  float level_db = 20.0f*log10f((float)level / (float)((1 << q_format) - 1));
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
  float level_db = 10.0f*log10f((float)level / (float)((1 << q_format) - 1));
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
    mant = _positive_float2fixed_saturate(alpha, Q_alpha);
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


/**
 * @brief Convert a graphic equaliser gain in decibels to a fixed point
 *        int32 number in Q31 format. The input level is shifted by -12 dB.
 *        This means that all the graphic EQ sliders can be set to +12
 *        without clipping, at the cost of -12dB level when the slider
 *        gains are set to 0dB.
 * 
 * @param level_db Level in db
 * @return int32_t level_db as an int32_t
 */
static inline int32_t geq_db_to_gain(float level_db) {
  return db_to_qxx(level_db - 12, 31);
}