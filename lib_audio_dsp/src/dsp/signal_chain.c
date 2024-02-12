
#include "dsp/adsp.h"

static inline int32_t f32_to_fixed(float x){
  float_s32_t v = f32_to_float_s32(x);
  right_shift_t shr = SIG_EXP - v.exp;
  asm("ashr %0, %1, %2": "=r" (v.mant): "r" (v.mant), "r" (shr));
  return v.mant;
}

int32_t adsp_dB_to_gain(float dB_gain) {
  float gain_fl = powf(10, (dB_gain / 20));
  return f32_to_fixed(gain_fl);
}

int32_t adsp_from_q31(int32_t input) {
  right_shift_t shr = 31 + SIG_EXP;
  asm("ashr %0, %1, %2": "=r" (input): "r" (input), "r" (shr));
  return input;
}

int32_t adsp_to_q31(int32_t input) {
  // put input into ah so that exp = sig_exp - 32
  // and then to get -31 saturate and extract with sig_exp - 1
  int32_t ah = input, al = 0, pos = -SIG_EXP + 1;
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (pos), "0" (ah), "1" (al));
  asm("lextract %0,%1,%2,%3,32":"=r"(ah):"r"(ah),"r"(al),"r"(pos));
  return ah;
}

int32_t adsp_adder(int32_t * input, unsigned n_ch) {
  // there is a bug in lsats, so it doesn't work if we give it 0,
  // so mul all samples by 2 and then lsats and lextract with 1
  int32_t ah = 0, al = 0, mul = 2, one = 1;
  for (unsigned i = 0; i < n_ch; i ++) {
    asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (input[i]), "r" (mul), "0" (ah), "1" (al));
  }
  // make sure we didn't overflow 32 bits
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (one), "0" (ah), "1" (al));
  asm("lextract %0 ,%1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (one));
  return ah;
}

int32_t adsp_subtractor(int32_t in1, int32_t in2) {
  // there is a bug in lsats, so it doesn't work if we give it 0,
  // so mul all samples by 2 and then lsats and lextract with 1
  int32_t ah = 0, al = 0, mul = 2, one = 1;
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (in1), "r" (mul), "0" (ah), "1" (al));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (in2), "r" (-mul), "0" (ah), "1" (al));
  // make sure we didn't overflow 32 bits
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (one), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (one));
  return ah;
}

int32_t adsp_fixed_gain(int32_t input, int32_t gain) {
  int32_t ah, al;
  int32_t q_format = -SIG_EXP;
  // adding 1 << (q_format -1) for rounding
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (input), "r" (gain), "0" (0), "1" (1 << (q_format - 1)));
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (q_format), "0" (ah), "1" (al));
  asm("lextract %0 ,%1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (q_format));
  return ah;
}

int32_t adsp_mixer(int32_t * input, int32_t gain, unsigned n_ch) {
  // there is a bug in lsats, so it doesn't work if we give it 0,
  // so mul all samples by 2 and then lsats and lextract with 1
  int32_t ah = 0, al = 0, mul = 2, one = 1;
  for (unsigned i = 0; i < n_ch; i ++) {
    int32_t tmp = adsp_fixed_gain(input[i], gain);
    asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (tmp), "r" (mul), "0" (ah), "1" (al));
  }
  // make sure we didn't overflow 32 bits
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (one), "0" (ah), "1" (al));
  asm("lextract %0 ,%1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (one));
  return ah;
}

