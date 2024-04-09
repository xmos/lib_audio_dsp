// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#define Q_alpha (31)

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
