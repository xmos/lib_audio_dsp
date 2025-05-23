// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <string.h>
#include <xcore/assert.h>
#include "dsp/adsp.h"
#include "dsp/_helpers/generic_utils.h"
#include "dsp/_helpers/reverb_utils.h"

#define DEFAULT_AP_LENS {142, 107, 379, 277, 2656, 1800}
#define DEFAULT_DELAY_LENS {4217, 4453, 3136, 3720}
#define DEFAULT_MOD_AP_LENS {908, 672}

#define DEFAULT_TAPS_L {266, 2974, 1913, 1996, 1990, 187, 1066}
#define DEFAULT_TAPS_R {353, 3627, 1228, 2673, 2111, 335, 121}

// EFFECT_GAIN is this reverb's "makeup gain". It is applied to the wet signal
// after the wet gain. When set properly, the makeup gain should have the effect
// of bringing the wet signal level up to match the dry signal, assuming the wet 
// and dry gains are equal.
//
// This hardcoded value of -1dB was found to be correct for the default config.
// It is set here and not by the user via wet-gain as it could be out of range of the
// Q31 wet gain configuration parameter. Possible future enhancement: make configurable.
#define EFFECT_GAIN 119621676 // -1 dB linear in q27
#if Q_GAIN != 27
#error "Need to change the EFFECT_GAIN"
#endif

#define TWO_TO_31 0x80000000

typedef struct
{
  void *heap_start;
  uint32_t num_bytes;
  uint32_t allocated_bytes;
} mem_manager_t;

static inline mem_manager_t mem_manager_init(
  void *heap_address,
  uint32_t number_of_bytes)
{
  mem_manager_t mem;
  mem.heap_start = heap_address;
  mem.num_bytes = number_of_bytes;
  mem.allocated_bytes = 0;
  return mem;
}

static inline void *mem_manager_alloc(mem_manager_t *mem, size_t size)
{
  // The xcore is byte aligned! But some instructions require word aligned
  // or even double-word aligned data to operate correctly. This allocator
  // has no opinions about this - caveat emptor.
  void *ret_address = NULL;
  if (size <= (mem->num_bytes - mem->allocated_bytes))
  {
    ret_address = mem->heap_start + mem->allocated_bytes;
    mem->allocated_bytes += size;
  }
  return ret_address;
}

lowpass_1ord_t lowpass_1ord_init(int32_t feedback) {
  lowpass_1ord_t lp;
  lp.filterstore = 0;
  feedback = (feedback < 1) ? 1 : feedback;
  lp.damp_1 = feedback;
  lp.damp_2 = (uint32_t)TWO_TO_31 - feedback;
  return lp;
}

static inline int32_t lowpass_1ord(lowpass_1ord_t * lp, int32_t new_samp) {
  /*
  output = (sample * self.coeff_b0_int) + (self._filterstore * self.coeff_a1_int)
  self._filterstore = output
  return output
  */

  int32_t ah = 0, al = 0, shift = Q_RVP;
  int32_t fstore = lp->filterstore, d1 = lp->damp_1, d2 = lp->damp_2;

  asm volatile("maccs %0, %1, %2, %3"
               : "=r"(ah), "=r"(al)
               : "r"(new_samp), "r"(d1), "0"(ah), "1"(al));
  asm volatile("maccs %0, %1, %2, %3"
               : "=r"(ah), "=r"(al)
               : "r"(fstore), "r"(d2), "0"(ah), "1"(al));

  al = scale_sat_int64_to_int32_floor(ah, al, shift);
  lp->filterstore = al;
  return al;
}

static inline allpass_fv_t allpass_fv_init(
  uint32_t max_delay,
  int32_t feedback_gain,
  void * mem)
{
  xassert(mem != NULL);
  allpass_fv_t ap;
  ap.max_delay = max_delay;
  ap.delay = max_delay;
  ap.feedback = feedback_gain;
  ap.buffer_idx = 0;
  ap.buffer = mem;
  return ap;
}

int32_t allpass_2(allpass_fv_t * ap, int32_t new_samp) {
  /*
  buf_out <- buf
  buf_in = new_samp - (buf_out * fb)

  buf <- buf_in
  inc buf

  out = buf_out + new_samp * fb
  return out
  */
  int32_t buf_out = ap->buffer[ap->buffer_idx];
  int32_t buf_in = add_with_fb(new_samp, buf_out, -ap->feedback, Q_RVP);

  ap->buffer[ap->buffer_idx] = buf_in;
  ap->buffer_idx += 1;
  if (ap->buffer_idx >= ap->delay)
  {
    ap->buffer_idx = 0;
  }

  int32_t output = add_with_fb(buf_out, buf_in, ap->feedback, Q_RVP);
  return output;
}

// 0.6 * 2 ** 29 = 322122547.2
// chosen as one of the closest ot the whole number
// and to give extra headroom for maccs
#define SCALE 322122547
#define SCALE_Q 29

void adsp_reverb_plate(
  reverb_plate_t *rv,
  int32_t outputs_lr[2],
  int32_t in_left,
  int32_t in_right)
{
  int32_t reverb_input = mix_with_pregain(in_left, in_right, rv->pre_gain, Q_RVP);
  reverb_input = adsp_delay(&rv->predelay, reverb_input);
  reverb_input = lowpass_1ord(&rv->lowpasses[0], reverb_input);

  for (unsigned i = 0; i < 4; i++) {
    reverb_input = allpass_2(&rv->allpasses[i], reverb_input);
  }

  int32_t path_1 = reverb_input, path_2 = reverb_input;
  path_1 = add_with_fb(path_1, rv->paths[0], rv->decay, Q_RVP);
  path_2 = add_with_fb(path_2, rv->paths[1], rv->decay, Q_RVP);

  path_1 = allpass_2(&rv->mod_allpasses[0], path_1);
  path_1 = adsp_delay(&rv->delays[0], path_1);
  path_1 = lowpass_1ord(&rv->lowpasses[1], path_1);
  path_1 = apply_gain_q31(path_1, rv->decay);
  path_1 = allpass_2(&rv->allpasses[4], path_1);
  rv->paths[1] = adsp_delay(&rv->delays[1], path_1);

  path_2 = allpass_2(&rv->mod_allpasses[1], path_2);
  path_2 = adsp_delay(&rv->delays[2], path_2);
  path_2 = lowpass_1ord(&rv->lowpasses[2], path_2);
  path_2 = apply_gain_q31(path_2, rv->decay);
  path_2 = allpass_2(&rv->allpasses[5], path_2);
  rv->paths[0] = adsp_delay(&rv->delays[3], path_2);

  int32_t out_l, out_r, q = SCALE_Q, ah = 0, al = 1 << (q - 1);

  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (rv->delays[0].buffer[rv->taps_l[0]]), "r" (SCALE), "0" (ah), "1" (al));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (rv->delays[0].buffer[rv->taps_l[1]]), "r" (SCALE), "0" (ah), "1" (al));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (rv->allpasses[4].buffer[rv->taps_l[2]]), "r" (-SCALE), "0" (ah), "1" (al));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (rv->delays[1].buffer[rv->taps_l[3]]), "r" (SCALE), "0" (ah), "1" (al));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (rv->delays[2].buffer[rv->taps_l[4]]), "r" (-SCALE), "0" (ah), "1" (al));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (rv->allpasses[5].buffer[rv->taps_l[5]]), "r" (-SCALE), "0" (ah), "1" (al));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (rv->delays[3].buffer[rv->taps_l[6]]), "r" (-SCALE), "0" (ah), "1" (al));

  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (q), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (out_l): "r" (ah), "r" (al), "r" (q));

  ah = 0; al = 1 << (q - 1);
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (rv->delays[2].buffer[rv->taps_r[0]]), "r" (SCALE), "0" (ah), "1" (al));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (rv->delays[2].buffer[rv->taps_r[1]]), "r" (SCALE), "0" (ah), "1" (al));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (rv->allpasses[5].buffer[rv->taps_r[2]]), "r" (-SCALE), "0" (ah), "1" (al));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (rv->delays[3].buffer[rv->taps_r[3]]), "r" (SCALE), "0" (ah), "1" (al));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (rv->delays[0].buffer[rv->taps_r[4]]), "r" (-SCALE), "0" (ah), "1" (al));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (rv->allpasses[4].buffer[rv->taps_r[5]]), "r" (-SCALE), "0" (ah), "1" (al));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (rv->delays[1].buffer[rv->taps_r[6]]), "r" (-SCALE), "0" (ah), "1" (al));

  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (q), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (out_r): "r" (ah), "r" (al), "r" (q));

  for (unsigned i = 0; i < ADSP_RVP_N_OUT_TAPS; i++) {
    rv->taps_l[i]++;
    rv->taps_r[i]++;
    if (rv->taps_l[i] >= rv->taps_len_l[i]) {
      rv->taps_l[i] = 0;
    }
    if (rv->taps_r[i] >= rv->taps_len_r[i]) {
      rv->taps_r[i] = 0;
    }
  }

  int32_t out = adsp_crossfader(out_l, out_r, rv->wet_gain1, rv->wet_gain2, Q_RVP);
  out = mix_wet_dry(out, apply_gain_q31(in_left, rv->dry_gain), EFFECT_GAIN, Q_GAIN);
  outputs_lr[0] = out;

  out = adsp_crossfader(out_r, out_l, rv->wet_gain1, rv->wet_gain2, Q_RVP);
  out = mix_wet_dry(out, apply_gain_q31(in_right, rv->dry_gain), EFFECT_GAIN, Q_GAIN);
  outputs_lr[1] = out;
}

void adsp_reverb_plate_init_filters(
  reverb_plate_t * rv,
  float fs,
  int32_t decay_diffusion_1,
  int32_t decay_diffusion_2,
  int32_t in_diffusion_1,
  int32_t in_diffusion_2,
  uint32_t max_predelay, 
  uint32_t predelay, 
  void * reverb_heap)
{
  // init all memory stuff
  mem_manager_t mem = mem_manager_init(reverb_heap, ADSP_RVP_HEAP_SZ(fs, max_predelay));
  const float rv_scale_fac = ADSP_RVP_SCALE(fs);

  uint32_t ap_lens[ADSP_RVP_N_APS] = DEFAULT_AP_LENS;
  uint32_t delay_lens[ADSP_RVP_N_DELAYS] = DEFAULT_DELAY_LENS;
  uint32_t mod_ap_lens[ADSP_RVP_N_PATHS] = DEFAULT_MOD_AP_LENS;
  memset(rv->paths, 0, ADSP_RVP_N_PATHS * sizeof(int32_t));

  // predelay
  void * delay_mem = mem_manager_alloc(&mem, DELAY_DSP_REQUIRED_MEMORY_SAMPLES(max_predelay));
  xassert(delay_mem != NULL);
  rv->predelay.fs = fs;
  rv->predelay.buffer_idx = 0;
  rv->predelay.delay = predelay;
  rv->predelay.max_delay = max_predelay;
  rv->predelay.buffer = delay_mem;

  // allpasses
  // yes, it makes me cry as well
  for (unsigned i = 0; i < 2; i++) {
    ap_lens[i] *= rv_scale_fac;
    void * ap_mem = mem_manager_alloc(&mem, ap_lens[i]<<2);
    rv->allpasses[i] = allpass_fv_init(ap_lens[i], in_diffusion_1, ap_mem);
  }
  for (unsigned i = 2; i < 4; i++) {
    ap_lens[i] *= rv_scale_fac;
    void * ap_mem = mem_manager_alloc(&mem, ap_lens[i]<<2);
    rv->allpasses[i] = allpass_fv_init(ap_lens[i], in_diffusion_2, ap_mem);
  }
  for (unsigned i = 4; i < 6; i++) {
    ap_lens[i] *= rv_scale_fac;
    void * ap_mem = mem_manager_alloc(&mem, ap_lens[i]<<2);
    rv->allpasses[i] = allpass_fv_init(ap_lens[i], decay_diffusion_2, ap_mem);
  }

  // mod allpasses
  for (unsigned i = 0; i < ADSP_RVP_N_PATHS; i++) {
    mod_ap_lens[i] *= rv_scale_fac;
    void * ap_mem = mem_manager_alloc(&mem, mod_ap_lens[i]<<2);
    rv->mod_allpasses[i] = allpass_fv_init(mod_ap_lens[i], decay_diffusion_1, ap_mem);
  }

  // delays
  for (unsigned i = 0; i < ADSP_RVP_N_DELAYS; i++) {
    delay_lens[i] *= rv_scale_fac;
    void * delay_mem = mem_manager_alloc(&mem, delay_lens[i]<<2);
    xassert(delay_mem != NULL);
    rv->delays[i].fs = fs;
    rv->delays[i].buffer_idx = 0;
    rv->delays[i].delay = delay_lens[i];
    rv->delays[i].max_delay = delay_lens[i];
    rv->delays[i].buffer = delay_mem;
  }

  // set tap lens
  rv->taps_len_l[0] = rv->delays[0].max_delay;
  rv->taps_len_l[1] = rv->delays[0].max_delay;
  rv->taps_len_l[2] = rv->allpasses[4].max_delay;
  rv->taps_len_l[3] = rv->delays[1].max_delay;
  rv->taps_len_l[4] = rv->delays[2].max_delay;
  rv->taps_len_l[5] = rv->allpasses[5].max_delay;
  rv->taps_len_l[6] = rv->delays[3].max_delay;

  rv->taps_len_r[0] = rv->delays[2].max_delay;
  rv->taps_len_r[1] = rv->delays[2].max_delay;
  rv->taps_len_r[2] = rv->allpasses[5].max_delay;
  rv->taps_len_r[3] = rv->delays[3].max_delay;
  rv->taps_len_r[4] = rv->delays[0].max_delay;
  rv->taps_len_r[5] = rv->allpasses[4].max_delay;
  rv->taps_len_r[6] = rv->delays[1].max_delay;

  memcpy(&rv->taps_l, (int32_t [] )DEFAULT_TAPS_L, sizeof(int32_t) * ADSP_RVP_N_OUT_TAPS);
  memcpy(&rv->taps_r, (int32_t [] )DEFAULT_TAPS_R, sizeof(int32_t) * ADSP_RVP_N_OUT_TAPS);

  for (unsigned i = 0; i < ADSP_RVP_N_OUT_TAPS; i ++) {
    rv->taps_l[i] *= rv_scale_fac;
    rv->taps_l[i] = rv->taps_len_l[i] - rv->taps_l[i];
    rv->taps_r[i] *= rv_scale_fac;
    rv->taps_r[i] = rv->taps_len_r[i] - rv->taps_r[i];
  }
}
