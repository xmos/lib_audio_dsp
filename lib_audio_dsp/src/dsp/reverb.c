// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <xcore/assert.h>
#include <string.h>

#include "dsp/adsp.h"
#include "dsp/_helpers/generic_utils.h"

#define DEFAULT_COMB_LENS                              \
    {                                                  \
        1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617 \
    }
#define DEFAULT_AP_LENS    \
    {                      \
        556, 441, 341, 225 \
    }

#define TWO_TO_31 0x80000000
#define TWO_TO_31_MINUS_1 0x7FFFFFFF

#define DEFAULT_SPREAD 23

#define DEFAULT_AP_FEEDBACK 0x40000000 // 0.5 in Q0.31

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

/**
 *
 * mem_manager class and methods
 *
 */

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

/**
 *
 * allpass_fv class and methods
 *
 */

static inline allpass_fv_t allpass_fv_init(
    uint32_t max_delay,
    int32_t feedback_gain,
    mem_manager_t *mem)
{
    allpass_fv_t ap;
    ap.max_delay = max_delay;
    ap.delay = 0;
    ap.feedback = feedback_gain;
    ap.buffer_idx = 0;
    ap.buffer = mem_manager_alloc(mem, max_delay * sizeof(int32_t));
    xassert(ap.buffer != NULL);
    return ap;
}

static inline void allpass_fv_set_delay(allpass_fv_t *ap, uint32_t delay)
{
    // saturate at max_delay
    ap->delay = (delay < ap->max_delay) ? delay : ap->max_delay;
}

static inline void allpass_fv_reset_state(allpass_fv_t *ap)
{
    memset(ap->buffer, 0, ap->max_delay * sizeof(int32_t));
}

int32_t allpass_fv(allpass_fv_t *ap, int32_t new_sample)
{
    int32_t ah = 0, al = 0, shift = Q_RVR;
    int32_t buf_out = ap->buffer[ap->buffer_idx];

    // Do (buf_out - new_sample) and saturate
    int32_t retval = adsp_subtractor(buf_out, new_sample);

    // Do (new_sample << shift) into a double word ah:al
    int64_t a = (int64_t)new_sample << shift;
    ah = (int32_t)(a >> 32);
    al = (int32_t)a;
    // Then do ah:al + (buf_out * feedback)
    asm volatile("maccs %0, %1, %2, %3"
                 : "=r"(ah), "=r"(al)
                 : "r"(buf_out), "r"(ap->feedback), "0"(ah), "1"(al));

    ap->buffer[ap->buffer_idx] = scale_sat_int64_to_int32_floor(ah, al, shift);
    ap->buffer_idx += 1;
    if (ap->buffer_idx >= ap->delay)
    {
        ap->buffer_idx = 0;
    }

    return retval;
}

/**
 *
 * comb_fv class and methods
 *
 */

static inline comb_fv_t comb_fv_init(
    uint32_t max_delay,
    int32_t feedback_gain,
    int32_t damping,
    mem_manager_t *mem)
{
    comb_fv_t comb;
    comb.max_delay = max_delay;
    comb.delay = 0;
    comb.feedback = feedback_gain;
    comb.buffer_idx = 0;
    comb.filterstore = 0;
    // damping is always at least 1, because we guarantee this earlier
    comb.damp_1 = damping;
    comb.damp_2 = (uint32_t)TWO_TO_31 - damping;
    comb.buffer = mem_manager_alloc(mem, max_delay * sizeof(int32_t));
    xassert(comb.buffer != NULL);
    return comb;
}

static inline void comb_fv_set_delay(comb_fv_t *comb, uint32_t new_delay)
{
    // saturate at max_delay
    comb->delay = (new_delay < comb->max_delay) ? new_delay : comb->max_delay;
}

static inline void comb_fv_reset_state(comb_fv_t *comb)
{
    memset(comb->buffer, 0, comb->max_delay * sizeof(int32_t));
    comb->filterstore = 0;
}

static inline int32_t comb_fv(comb_fv_t *comb, int32_t new_sample)
{
    int32_t ah = 0, al = 0, shift = Q_RVR;
    int32_t fstore = comb->filterstore, d1 = comb->damp_1, d2 = comb->damp_2;
    int32_t output = comb->buffer[comb->buffer_idx];

    // Do (output * damp_2) into a 64b word ah:al
    asm volatile("maccs %0, %1, %2, %3"
                 : "=r"(ah), "=r"(al)
                 : "r"(output), "r"(d2), "0"(ah), "1"(al));
    // Then add (filterstore * damp_1) to that
    asm volatile("maccs %0, %1, %2, %3"
                 : "=r"(ah), "=r"(al)
                 : "r"(fstore), "r"(d1), "0"(ah), "1"(al));

    comb->filterstore = scale_sat_int64_to_int32_floor(ah, al, shift);
    fstore = comb->filterstore;

    // Do (new_sample << shift) into a 64b word ah:al
    int64_t a = (int64_t)new_sample << shift;
    ah = (int32_t)(a >> 32);
    al = (int32_t)a;

    // Then do ah:al + (fstore * feedback)
    asm volatile("maccs %0, %1, %2, %3"
                 : "=r"(ah), "=r"(al)
                 : "r"(fstore), "r"(comb->feedback), "0"(ah), "1"(al));

    comb->buffer[comb->buffer_idx] = scale_sat_int64_to_int32_floor(ah, al,
                                                                    shift);
    comb->buffer_idx += 1;
    if (comb->buffer_idx >= comb->delay)
    {
        comb->buffer_idx = 0;
    }

    return output;
}

void adsp_reverb_room_init_filters(
    reverb_room_t *rv,
    float fs,
    float max_room_size,
    uint32_t max_predelay,
    uint32_t predelay,
    int32_t feedback,
    int32_t damping,
    void * reverb_heap)
{
    mem_manager_t memory_manager = mem_manager_init(
        reverb_heap,
        ADSP_RVR_HEAP_SZ(fs, max_room_size, max_predelay));

    const float rv_scale_fac = ADSP_RVR_SCALE(fs, max_room_size);

    // shift 2 insted of / sizeof(int32_t), to avoid division
    rv->total_buffer_length = ADSP_RVR_HEAP_SZ(fs, max_room_size, max_predelay) >> 2;

    uint32_t comb_lengths[ADSP_RVR_N_COMBS] = DEFAULT_COMB_LENS;
    uint32_t ap_lengths[ADSP_RVR_N_APS] = DEFAULT_AP_LENS;
    for (int i = 0; i < ADSP_RVR_N_COMBS; i++)
    {
        // Scale maximum lengths by the scale factor (fs/44100 * max_room)
        comb_lengths[i] *= rv_scale_fac;
        rv->combs[i] = comb_fv_init(
            comb_lengths[i],
            feedback,
            damping,
            &memory_manager);
    }
    for (int i = 0; i < ADSP_RVR_N_APS; i++)
    {
        // Scale maximum lengths by the scale factor (fs/44100 * max_room)
        ap_lengths[i] *= rv_scale_fac;
        rv->allpasses[i] = allpass_fv_init(
            ap_lengths[i],
            DEFAULT_AP_FEEDBACK,
            &memory_manager);
    }
    // init predelay manually with memory_manager
    rv->predelay.fs = fs;
    rv->predelay.buffer_idx = 0;
    rv->predelay.delay = predelay;
    rv->predelay.max_delay = max_predelay;
    rv->predelay.buffer = mem_manager_alloc(&memory_manager, DELAY_DSP_REQUIRED_MEMORY_SAMPLES(max_predelay));
}

void adsp_reverb_room_reset_state(reverb_room_t *rv)
{
    for (int comb = 0; comb < ADSP_RVR_N_COMBS; comb++)
    {
        comb_fv_reset_state(&rv->combs[comb]);
    }
    for (int ap = 0; ap < ADSP_RVR_N_APS; ap++)
    {
        allpass_fv_reset_state(&rv->allpasses[ap]);
    }
}

uint32_t adsp_reverb_room_get_buffer_lens(reverb_room_t *rv)
{
    return rv->total_buffer_length;
}

void adsp_reverb_room_set_room_size(reverb_room_t *rv,
                                    float new_room_size)
{
    // For larger rooms, increase max_room_size
    xassert(new_room_size >= 0 && new_room_size <= 1);
    rv->room_size = new_room_size;
    // could use uq32 for the room size, but it's important
    // to represent 1.0 here, so loosing one bit of precision
    // and doing extra lextracts :(
    int32_t zero = 0, q = 31, exp;
    uint32_t ah = 0, al = 0, room_size_int;
    asm("fsexp %0, %1, %2": "=r"(zero), "=r"(exp): "r"(new_room_size));
    asm("fmant %0, %1": "=r"(room_size_int): "r"(new_room_size));
    right_shift_t shr = -q - exp + 23;
    room_size_int >>= shr;

    for (int comb = 0; comb < ADSP_RVR_N_COMBS; comb++)
    {
        // Do comb length * new_room_size in UQ0.31
        uint32_t l = rv->combs[comb].max_delay;
        asm volatile("lmul %0, %1, %2, %3, %4, %5"
                     : "=r"(ah), "=r"(al)
                     : "r"(room_size_int), "r"(l), "r"(zero), "r"(zero));
        asm volatile("lextract %0, %1, %2, %3, 32"
                     : "=r"(ah)
                     : "r"(ah), "r"(al), "r"(q));
        comb_fv_set_delay(&rv->combs[comb], ah);
    }
    for (int ap = 0; ap < ADSP_RVR_N_APS; ap++)
    {
        // Do ap length * new_room_size in UQ0.31
        uint32_t l = rv->allpasses[ap].max_delay;
        asm volatile("lmul %0, %1, %2, %3, %4, %5"
                     : "=r"(ah), "=r"(al)
                     : "r"(room_size_int), "r"(l), "r"(zero), "r"(zero));
        asm volatile("lextract %0, %1, %2, %3, 32"
                     : "=r"(ah)
                     : "r"(ah), "r"(al), "r"(q));
        allpass_fv_set_delay(&rv->allpasses[ap], ah);
    }
}

int32_t adsp_reverb_room(
    reverb_room_t *rv,
    int32_t new_samp)
{
    int32_t delayed_input = adsp_delay(&rv->predelay, new_samp);
    int32_t reverb_input = apply_gain_q31(delayed_input, rv->pre_gain);
    int32_t output = 0;
    int64_t acc = 0;
    for (int comb = 0; comb < ADSP_RVR_N_COMBS; comb++)
    {
        acc += comb_fv(&(rv->combs[comb]), reverb_input);
    }
    output = adsp_saturate_32b(acc);

    for (int ap = 0; ap < ADSP_RVR_N_APS; ap++)
    {
        acc = allpass_fv(&(rv->allpasses[ap]), output);
        output = (int32_t)acc; // We do not saturate here!
    }
    output = apply_gain_q31(output, rv->wet_gain);
    acc = adsp_fixed_gain(output, EFFECT_GAIN);
    acc += apply_gain_q31(new_samp, rv->dry_gain);
    output = adsp_saturate_32b(acc);

    return output;
}

void adsp_reverb_room_st_init_filters(
    reverb_room_st_t *rv,
    float fs,
    float max_room_size,
    uint32_t max_predelay,
    uint32_t predelay,
    int32_t feedback,
    int32_t damping,
    void * reverb_heap)
{
    mem_manager_t memory_manager = mem_manager_init(
        reverb_heap,
        ADSP_RVRST_HEAP_SZ(fs, max_room_size, max_predelay));

    const float rv_scale_fac = ADSP_RVR_SCALE(fs, max_room_size);

    // shift 2 insted of / sizeof(int32_t), to avoid division
    rv->total_buffer_length = ADSP_RVRST_HEAP_SZ(fs, max_room_size, max_predelay) >> 2;
    rv->spread_length = rv_scale_fac * DEFAULT_SPREAD;
    uint32_t comb_lengths[ADSP_RVR_N_COMBS] = DEFAULT_COMB_LENS;
    uint32_t ap_lengths[ADSP_RVR_N_APS] = DEFAULT_AP_LENS;
    for (int i = 0; i < ADSP_RVR_N_COMBS; i++)
    {
        // Scale maximum lengths by the scale factor (fs/44100 * max_room)
        comb_lengths[i] *= rv_scale_fac;
        rv->combs[0][i] = comb_fv_init(
            comb_lengths[i],
            feedback,
            damping,
            &memory_manager);
        rv->combs[1][i] = comb_fv_init(
            comb_lengths[i] + rv->spread_length,
            feedback,
            damping,
            &memory_manager);
    }
    for (int i = 0; i < ADSP_RVR_N_APS; i++)
    {
        // Scale maximum lengths by the scale factor (fs/44100 * max_room)
        ap_lengths[i] *= rv_scale_fac;
        rv->allpasses[0][i] = allpass_fv_init(
            ap_lengths[i],
            DEFAULT_AP_FEEDBACK,
            &memory_manager);
        rv->allpasses[1][i] = allpass_fv_init(
            ap_lengths[i] + rv->spread_length,
            DEFAULT_AP_FEEDBACK,
            &memory_manager);
    }
    // init predelay manually with memory_manager
    rv->predelay.fs = fs;
    rv->predelay.buffer_idx = 0;
    rv->predelay.delay = predelay;
    rv->predelay.max_delay = max_predelay;
    rv->predelay.buffer = mem_manager_alloc(&memory_manager, DELAY_DSP_REQUIRED_MEMORY_SAMPLES(max_predelay));
}

void adsp_reverb_room_st_set_room_size(reverb_room_st_t *rv,
                                    float new_room_size)
{
    // For larger rooms, increase max_room_size
    xassert(new_room_size >= 0 && new_room_size <= 1);
    rv->room_size = new_room_size;
    // could use uq32 for the room size, but it's important
    // to represent 1.0 here, so loosing one bit of precision
    // and doing extra lextracts :(
    int32_t zero = 0, q = 31, exp;
    uint32_t ah = 0, al = 0, room_size_int, sp_delay;
    asm("fsexp %0, %1, %2": "=r"(zero), "=r"(exp): "r"(new_room_size));
    asm("fmant %0, %1": "=r"(room_size_int): "r"(new_room_size));
    right_shift_t shr = -q - exp + 23;
    room_size_int >>= shr;

    asm("lmul %0, %1, %2, %3, %4, %5"
        : "=r" (ah), "=r" (al)
        : "r" (room_size_int), "r" (rv->spread_length), "r" (zero), "r" (zero));
    asm("lextract %0, %1, %2, %3, 32"
        : "=r" (sp_delay)
        : "r" (ah), "r" (al), "r" (q));

    for (int comb = 0; comb < ADSP_RVR_N_COMBS; comb++)
    {
        // Do comb length * new_room_size in UQ0.31
        uint32_t l = rv->combs[0][comb].max_delay;
        asm volatile("lmul %0, %1, %2, %3, %4, %5"
                     : "=r"(ah), "=r"(al)
                     : "r"(room_size_int), "r"(l), "r"(zero), "r"(zero));
        asm volatile("lextract %0, %1, %2, %3, 32"
                     : "=r"(ah)
                     : "r"(ah), "r"(al), "r"(q));
        comb_fv_set_delay(&rv->combs[0][comb], ah);
        comb_fv_set_delay(&rv->combs[1][comb], ah + sp_delay);
    }
    for (int ap = 0; ap < ADSP_RVR_N_APS; ap++)
    {
        // Do ap length * new_room_size in UQ0.31
        uint32_t l = rv->allpasses[0][ap].max_delay;
        asm volatile("lmul %0, %1, %2, %3, %4, %5"
                     : "=r"(ah), "=r"(al)
                     : "r"(room_size_int), "r"(l), "r"(zero), "r"(zero));
        asm volatile("lextract %0, %1, %2, %3, 32"
                     : "=r"(ah)
                     : "r"(ah), "r"(al), "r"(q));
        allpass_fv_set_delay(&rv->allpasses[0][ap], ah);
        allpass_fv_set_delay(&rv->allpasses[1][ap], ah + sp_delay);
    }
}

static inline int32_t _get_stereo_out(reverb_room_st_t *rv, int32_t out1, int32_t out2, int32_t input) {
    int32_t q = 31, ah = 0, al = 1 << (q - 1), wet_sig;
    asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (out1), "r" (rv->wet_gain1), "0" (ah), "1" (al));
    asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (out2), "r" (rv->wet_gain2), "0" (ah), "1" (al));
    asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (q), "0" (ah), "1" (al));
    asm("lextract %0, %1, %2, %3, 32": "=r" (wet_sig): "r" (ah), "r" (al), "r" (q));
    int64_t acc;
    acc = adsp_fixed_gain(wet_sig, EFFECT_GAIN);
    acc += apply_gain_q31(input, rv->dry_gain);
    int32_t output = adsp_saturate_32b(acc);
    return output;
}

void adsp_reverb_room_st(
    reverb_room_st_t *rv,
    int32_t outputs_lr[2],
    int32_t in_left,
    int32_t in_right)
{
    int32_t q = 31, ah = 0, al = 1 << (q - 1), reverb_input;
    asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (in_left), "r" (rv->pre_gain), "0" (ah), "1" (al));
    asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (in_right), "r" (rv->pre_gain), "0" (ah), "1" (al));
    asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (q), "0" (ah), "1" (al));
    asm("lextract %0, %1, %2, %3, 32": "=r" (reverb_input): "r" (ah), "r" (al), "r" (q));

    int32_t delayed_input = adsp_delay(&rv->predelay, reverb_input);
    int32_t out_r = 0, out_l = 0;
    int64_t acc_r = 0, acc_l = 0;

    for (int comb = 0; comb < ADSP_RVR_N_COMBS; comb++)
    {
        acc_l += comb_fv(&(rv->combs[0][comb]), delayed_input);
        acc_r += comb_fv(&(rv->combs[1][comb]), delayed_input);
    }
    out_l = adsp_saturate_32b(acc_l);
    out_r = adsp_saturate_32b(acc_r);

    for (int ap = 0; ap < ADSP_RVR_N_APS; ap++)
    {
        // We do not saturate here!
        acc_l = allpass_fv(&(rv->allpasses[0][ap]), out_l);
        out_l = (int32_t)acc_l;
        acc_r = allpass_fv(&(rv->allpasses[1][ap]), out_r);
        out_r = (int32_t)acc_r;
    }

    outputs_lr[0] = _get_stereo_out(rv, out_l, out_r, in_left);
    outputs_lr[1] = _get_stereo_out(rv, out_r, out_l, in_right);
}
