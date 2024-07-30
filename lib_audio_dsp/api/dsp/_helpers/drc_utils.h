// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "generic_utils.h"

/**
 * @brief Exponential moving average with a Q0.31 alpha
 * Upadates the exponential moving average of x with the new sample y
 * 
 * @param x         Current moving average
 * @param y         New sample
 * @param alpha     Exponential decay factor
 * @return int32_t  Updated moving average
 */
static inline int32_t q31_ema(int32_t x, int32_t y, q1_31 alpha) {
  // this assumes that x and y are positive and alpha is q31
  // x and y have to have the same exponent
  int32_t ah, al;
  int32_t mul = y - x;

  // preload the acc with x at position of 31
  // (essentially giving it exponent of -31 + x.exp)
  asm("linsert %0, %1, %2, %3, 32":"=r" (ah), "=r" (al): "r"(x), "r"(Q_alpha), "0"(0), "1" (0));
  // x + alpha * (y - x) with exponent -31 + x.exp
  asm("maccs %0,%1,%2,%3":"=r"(ah),"=r"(al):"r"(alpha),"r"(mul), "0" (ah), "1" (al));
  // saturate and extract from 63rd bit
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (Q_alpha), "0" (ah), "1" (al));
  asm("lextract %0,%1,%2,%3,32":"=r"(x):"r"(ah),"r"(al),"r"(Q_alpha));
  return x;
}

/**
 * @brief Positive IEEE 754 float to Q_sig conversion
 * 
 * @param val       Positive IEEE 754 float
 * @return int32_t  Q_SIG fixed-point
 */
static inline int32_t from_float_pos(float val) {
  // assumes that val is positive
  int32_t sign, exp, mant;
  asm("fsexp %0, %1, %2": "=r" (sign), "=r" (exp): "r" (val));
  asm("fmant %0, %1": "=r" (mant): "r" (val));
  // mant to SIG_EXP
  right_shift_t shr = SIG_EXP - exp + 23;
  mant >>= shr;
  return mant;
}
