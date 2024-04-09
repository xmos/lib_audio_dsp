// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "generic_utils.h"

static inline int32_t q31_ema(int32_t x, int32_t samp, q1_31 alpha) {
  // this assumes that x and samp are positive and alpha is q31
  // x and samp have to have the same exponent
  int32_t ah, al;
  int32_t mul = samp - x;

  // preload the acc with x at position of 31
  // (essentially giving it exponent of -31 + x.exp)
  asm("linsert %0, %1, %2, %3, 32":"=r" (ah), "=r" (al): "r"(x), "r"(Q_alpha), "0"(0), "1" (0));
  // x + alpha * (samp - x) with exponent -31 + x.exp
  asm("maccs %0,%1,%2,%3":"=r"(ah),"=r"(al):"r"(alpha),"r"(mul), "0" (ah), "1" (al));
  // saturate and extract from 63rd bit
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (Q_alpha), "0" (ah), "1" (al));
  asm("lextract %0,%1,%2,%3,32":"=r"(x):"r"(ah),"r"(al),"r"(Q_alpha));
  return x;
}
