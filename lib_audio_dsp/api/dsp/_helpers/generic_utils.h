// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

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
