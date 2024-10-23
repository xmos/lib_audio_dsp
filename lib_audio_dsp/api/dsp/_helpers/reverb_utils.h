// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

static inline int32_t scale_sat_int64_to_int32_floor(int32_t ah,
                                                     int32_t al,
                                                     int32_t shift)
{
  int32_t big_q = ((uint32_t)1 << shift) - 1, one = 1, shift_minus_one = shift - 1;

  // If ah:al < 0, add just under 1 (represented in Q0.31)
  if (ah < 0) // ah is sign extended, so this test is sufficient
  {
    asm volatile("maccs %0, %1, %2, %3"
                 : "=r"(ah), "=r"(al)
                 : "r"(one), "r"(big_q), "0"(ah), "1"(al));
  }
  // Saturate ah:al. Implements the following:
  // if (val > (2 ** (31 + shift) - 1))
  //     val = 2 ** (31 + shift) - 1
  // else if (val < -(2 ** (31 + shift)))
  //     val = -(2 ** (31 + shift))
  // Note the use of 31, rather than 32 - hence here we subtract 1 from shift.
  asm volatile("lsats %0, %1, %2"
               : "=r"(ah), "=r"(al)
               : "r"(shift_minus_one), "0"(ah), "1"(al));
  // then we return (ah:al >> shift)
  asm volatile("lextract %0, %1, %2, %3, 32"
               : "=r"(ah)
               : "r"(ah), "r"(al), "r"(shift));

  return ah;
}

static inline int32_t add_with_fb(int32_t samp1, int32_t samp2, int32_t fb, int32_t shift) {
  // samp1 + (samp2 * feedback)
  int32_t ah, al;
  int64_t a = (int64_t)samp1 << shift;
  ah = (int32_t)(a >> 32);
  al = (int32_t)a;

  asm volatile("maccs %0, %1, %2, %3"
               : "=r"(ah), "=r"(al)
               : "r"(samp2), "r"(fb), "0"(ah), "1"(al));

  ah = scale_sat_int64_to_int32_floor(ah, al, shift);
  return ah;
}

static inline int32_t mix_with_pregain(int32_t sig1, int32_t sig2, int32_t pregain, int32_t q_gain) {
  int32_t ah = 0, al = 1 << (q_gain - 1);
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (sig1), "r" (pregain), "0" (ah), "1" (al));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (sig2), "r" (pregain), "0" (ah), "1" (al));
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (q_gain), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (q_gain));
  return ah;
}

// only exists because the effect gain is not a part of the wet gain, so need to handle properly
// will do: wet_sig * effect_gain + dry_sig
static inline int32_t mix_wet_dry(int32_t wet_sig, int32_t dry_sig, int32_t effect_gain, int32_t q_gain) {
  int32_t ah = 0, al = 1 << (q_gain - 1);
  asm("linsert %0, %1, %2, %3, 32": "=r" (ah), "=r" (al): "r" (dry_sig), "r" (q_gain), "0" (ah), "1" (al));
  asm("sext %0, %1": "=r" (ah): "r" (q_gain), "0" (ah));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (wet_sig), "r" (effect_gain), "0" (ah), "1" (al));
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (q_gain), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (q_gain));
  return ah;
}

// will do out1 * gain1 + out2 * gain2, assumes that both gains are q31
static inline int32_t mix_wet_chans(int32_t out1, int32_t out2, int32_t gain1, int32_t gain2, int32_t q_gain) {
  int32_t ah = 0, al = 1 << (q_gain - 1);
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (out1), "r" (gain1), "0" (ah), "1" (al));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (out2), "r" (gain2), "0" (ah), "1" (al));
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (q_gain), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (q_gain));
  return ah;
}
