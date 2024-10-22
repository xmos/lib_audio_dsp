#include <string.h>
#include <xcore/assert.h>
#include "dsp/adsp.h"
#include "dsp/_helpers/generic_utils.h"

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
// This hardcoded value of 10dB was found to be correct for the default config.
// It is set here and not by the user via wet-gain as it is out of range of the
// Q31 wet gain configuration parameter. Possible future enhancement: make configurable.
#define EFFECT_GAIN 424433723 // 10 dB linear in q27
#if Q_GAIN != 27
#error "Need to change the EFFECT_GAIN"
#endif

#define TWO_TO_31_MINUS_1 0x7FFFFFFF
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

static inline int32_t scale_sat_int64_to_int32_floor(int32_t ah,
                                                     int32_t al,
                                                     int32_t shift)
{
  int32_t big_q = TWO_TO_31_MINUS_1, one = 1, shift_minus_one = shift - 1;

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
  output = (sample * self.damp1) + (self._filterstore * self.damp2)
  self._filterstore = output
  return output
  */

  int32_t ah = 0, al = 0, shift = 31;
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

static inline int32_t add_with_fb(int32_t samp1, int32_t samp2, int32_t fb) {
  // samp1 + (samp2 * feedback)
  int32_t ah, al, shift = Q_RVR;
  int64_t a = (int64_t)samp1 << shift;
  ah = (int32_t)(a >> 32);
  al = (int32_t)a;

  asm volatile("maccs %0, %1, %2, %3"
               : "=r"(ah), "=r"(al)
               : "r"(samp2), "r"(fb), "0"(ah), "1"(al));

  al = scale_sat_int64_to_int32_floor(ah, al, shift);
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
  int32_t buf_in = add_with_fb(new_samp, buf_out, -ap->feedback);

  ap->buffer[ap->buffer_idx] = buf_in;
  ap->buffer_idx += 1;
  if (ap->buffer_idx >= ap->delay)
  {
    ap->buffer_idx = 0;
  }

  int32_t output = add_with_fb(buf_out, new_samp, ap->feedback);
  return output;
}

// only exists because the effect gain is not a part of the wet gain, so need to handle properly
// will do: wet_sig * effect_gain + dry_sig
static inline int32_t _mix_wet_dry(int32_t wet_sig, int32_t dry_sig){
  int32_t ah = 0, al = 0, q = Q_GAIN;
  asm("linsert %0, %1, %2, %3, 32": "=r" (ah), "=r" (al): "r" (dry_sig), "r" (q), "0" (ah), "1" (al));
  asm("sext %0, %1": "=r" (ah): "r" (q), "0" (ah));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (wet_sig), "r" (EFFECT_GAIN), "0" (ah), "1" (al));
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (q), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (q));
  return ah;
}

// will do out1 * gain1 + out2 * gain2, assumes that both gains are q31
static inline int32_t _get_wet_signal(int32_t out1, int32_t out2, int32_t gain1, int32_t gain2) {
  int32_t q = 31, ah = 0, al = 1 << (q - 1);
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (out1), "r" (gain1), "0" (ah), "1" (al));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (out2), "r" (gain2), "0" (ah), "1" (al));
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (q), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (q));
  return ah;
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
  int32_t q = 31, ah = 0, al = 1 << (q - 1), reverb_input;
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (in_left), "r" (rv->pre_gain), "0" (ah), "1" (al));
  asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (in_right), "r" (rv->pre_gain), "0" (ah), "1" (al));
  asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (q), "0" (ah), "1" (al));
  asm("lextract %0, %1, %2, %3, 32": "=r" (reverb_input): "r" (ah), "r" (al), "r" (q));

  reverb_input = adsp_delay(&rv->predelay, reverb_input);
  reverb_input = lowpass_1ord(&rv->lowpasses[0], reverb_input);

  for (unsigned i = 0; i < 4; i++) {
    reverb_input = allpass_2(&rv->allpasses[i], reverb_input);
  }

  int32_t path_1 = reverb_input, path_2 = reverb_input;
  path_1 = add_with_fb(path_1, rv->paths[0], rv->decay);
  path_2 = add_with_fb(path_2, rv->paths[1], rv->decay);

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

  int32_t out_l, out_r;
  q = SCALE_Q;
  ah = 0; al = 1 << (q - 1);
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

  int32_t out = _get_wet_signal(out_l, out_r, rv->wet_gain1, rv->wet_gain2);
  out = _mix_wet_dry(out, apply_gain_q31(in_left, rv->dry_gain));
  outputs_lr[0] = out;

  out = _get_wet_signal(out_r, out_l, rv->wet_gain1, rv->wet_gain2);
  out = _mix_wet_dry(out, apply_gain_q31(in_right, rv->dry_gain));
  outputs_lr[1] = out;
}

void adsp_reverb_plate_init_filters(
  reverb_plate_t * rv,
  float fs,
  int32_t decay,
  int32_t diffusion,
  int32_t in_diffusion_1,
  int32_t in_diffusion_2,
  uint32_t max_predelay, 
  uint32_t predelay, 
  void * reverb_heap)
{
  rv->paths[0] = 0;
  rv->paths[1] = 0;
  // init all memory stuff
  mem_manager_t mem = mem_manager_init(reverb_heap, ADSP_RVP_HEAP_SZ(fs, max_predelay));
  const float rv_scale_fac = ADSP_RVP_SCALE(fs);

  uint32_t ap_lens[ADSP_RVP_N_APS] = DEFAULT_AP_LENS;
  uint32_t delay_lens[ADSP_RVP_N_DELAYS] = DEFAULT_DELAY_LENS;
  uint32_t mod_ap_lens[ADSP_RVP_N_PATHS] = DEFAULT_MOD_AP_LENS;

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
    rv->allpasses[i] = allpass_fv_init(ap_lens[i], decay, ap_mem);
  }

  // mod allpasses
  for (unsigned i = 0; i < ADSP_RVP_N_PATHS; i++) {
    mod_ap_lens[i] *= rv_scale_fac;
    void * ap_mem = mem_manager_alloc(&mem, mod_ap_lens[i]<<2);
    rv->mod_allpasses[i] = allpass_fv_init(mod_ap_lens[i], diffusion, ap_mem);
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
