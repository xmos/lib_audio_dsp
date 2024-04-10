// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <xcore/assert.h>
#include <math.h>
#include <string.h>

#include "dsp/adsp.h"
#include "dsp/reverb.h"
#include "dsp/_helpers/generic_utils.h"

#define DEFAULT_COMB_LENS                              \
    {                                                  \
        1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617 \
    }
#define DEFAULT_AP_LENS    \
    {                      \
        556, 441, 341, 225 \
    }

#define DBTOGAIN(x) (powf(10, (x / 20.0)))
#define GAINTODB(x) (log10f(x) * 20.0)
#define TWO_TO_31_MINUS_1 2147483647

#define Q_RV 31
#define DEFAULT_PREGAIN 0.015f
#define DEFAULT_AP_FEEDBACK 0.5

static inline int32_t scale_sat_int64_to_int32_floor(int32_t ah,
                                                     int32_t al,
                                                     int32_t shift)
{
    int32_t big_q = TWO_TO_31_MINUS_1, one = 1, shift_minus_one = shift - 1;

    // If ah:al < 0, add just under 1 (represented in Q31)
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
    uint32_t starting_delay,
    int32_t feedback_gain,
    mem_manager_t *mem)
{
    allpass_fv_t ap;
    ap.max_delay = max_delay;
    ap.delay = starting_delay;
    ap.feedback = feedback_gain;
    ap.buffer_idx = 0;
    ap.buffer = mem_manager_alloc(mem, max_delay * sizeof(int32_t));
    xassert(ap.buffer != NULL);
    return ap;
}

static inline void allpass_fv_set_delay(allpass_fv_t *ap, uint32_t delay)
{
    if (delay < ap->max_delay)
    {
        ap->delay = delay;
    }
    else
    {
        ap->delay = ap->max_delay;
        // Delay cannot be greater than max delay, setting to max delay
    }
}

static inline void allpass_fv_reset_state(allpass_fv_t *ap)
{
    memset(ap->buffer, 0, ap->max_delay * sizeof(int32_t));
}

int32_t allpass_fv(allpass_fv_t *ap, int32_t new_sample)
{
    int32_t ah = 0, al = 0, shift = Q_RV;
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
    uint32_t starting_delay,
    int32_t feedback_gain,
    int32_t damping,
    mem_manager_t *mem)
{
    comb_fv_t comb;
    comb.max_delay = max_delay;
    comb.delay = starting_delay;
    comb.feedback = feedback_gain;
    comb.buffer_idx = 0;
    comb.filterstore = 0;
    comb.damp_1 = damping;
    comb.damp_2 = ((1 << 31) - 1) - damping + 1;
    comb.buffer = mem_manager_alloc(mem, max_delay * sizeof(int32_t));
    xassert(comb.buffer != NULL);
    return comb;
}

static inline void comb_fv_set_delay(comb_fv_t *comb, uint32_t new_delay)
{
    if (new_delay < comb->max_delay)
    {
        comb->delay = new_delay;
    }
    else
    {
        comb->delay = comb->max_delay;
        // Delay cannot be greater than max delay, setting to max delay
    }
}

static inline void comb_fv_reset_state(comb_fv_t *comb)
{
    memset(comb->buffer, 0, comb->max_delay * sizeof(int32_t));
    comb->filterstore = 0;
}

static inline int32_t comb_fv(comb_fv_t *comb, int32_t new_sample)
{
    int32_t ah = 0, al = 0, shift = Q_RV;
    int32_t fstore = comb->filterstore, d1 = comb->damp_1, d2 = comb->damp_2;
    int32_t retval = comb->buffer[comb->buffer_idx];

    // Do (retval * damp_2) into a 64b word ah:al
    asm volatile("maccs %0, %1, %2, %3"
                 : "=r"(ah), "=r"(al)
                 : "r"(retval), "r"(d2), "0"(ah), "1"(al));
    // Then add (filterstore * damp_1) to that
    asm volatile("maccs %0, %1, %2, %3"
                 : "=r"(ah), "=r"(al)
                 : "r"(fstore), "r"(d1), "0"(ah), "1"(al));

    comb->filterstore = scale_sat_int64_to_int32_floor(ah, al, shift);
    fstore = comb->filterstore;

    ah = 0;
    al = 0;

    // Do (new_sample << Q_RV) into a 64b word ah:al
    /*asm volatile("linsert %0, %1, %2, %3, 32"
                 : "=r"(ah), "=r"(al)
                 : "r"(new_sample), "r"(shift));*/

    int64_t a = (int64_t)new_sample << 31;
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

    return retval;
}

int32_t adsp_reverb_calc_wet_gain(float wet_gain_db, float pregain)
{
    xassert(wet_gain_db > MIN_WET_GAIN_DB && wet_gain_db <= MAX_WET_GAIN_DB);
    xassert(pregain > 4.66e-10 && pregain < 1);
    int32_t wet = Q(Q_RV)(DBTOGAIN(wet_gain_db)); // *
                                                  //(DEFAULT_PREGAIN / pregain));
    return wet;
}

int32_t adsp_reverb_calc_dry_gain(float dry_gain_db)
{
    xassert(dry_gain_db > -186 && dry_gain_db <= 0);
    int32_t dry = Q(Q_RV)(DBTOGAIN(dry_gain_db));
    return dry;
}

/**
 *
 * reverb_room class and methods
 *
 */

reverb_room_t adsp_reverb_room_init(
    float fs,
    float max_room_size,
    float room_size,
    float decay,
    float damping,
    int32_t wet_gain,
    int32_t dry_gain,
    float pregain,
    void *reverb_heap)
{
    // max_room_size should really be below 4 for reasons of good taste
    xassert(max_room_size > 0 && max_room_size <= MAX_ROOM_SIZE);
    // For larger rooms, increase max_room_size. Don't forget to also increase
    // the size of reverb_heap
    xassert(room_size >= 0 && room_size <= 1);
    xassert(decay >= 0 && decay <= 1);
    xassert(damping >= 0 && damping <= 1);

    // These limits should be reconsidered if Q_RV != 31
    // Represented as q1_31, min nonzero val 4.66e-10 ~= -186 dB
    xassert(pregain > 4.66e-10 && pregain < 1);

    mem_manager_t memory_manager = mem_manager_init(
        reverb_heap,
        RV_HEAP_SZ(fs, max_room_size));

    reverb_room_t rv;

    // Avoids too much or too little feedback
    const int32_t feedback_int = Q(Q_RV)((decay * 0.28) + 0.7);
    const int32_t damping_int = MAX(Q(Q_RV)(damping) - 1, 1);

    // Scale the wet gain; when pregain changes, overall wet gain shouldn't
    const float rv_scale_fac = RV_SCALE(fs, max_room_size);

    rv.total_buffer_length = RV_HEAP_SZ(fs, max_room_size) / sizeof(int32_t);
    rv.room_size = Q30(room_size);
    rv.dry_gain = dry_gain;
    rv.wet_gain = wet_gain;
    rv.pre_gain = Q(Q_RV)(pregain);

    int32_t comb_lengths[8] = DEFAULT_COMB_LENS;
    int32_t ap_lengths[4] = DEFAULT_AP_LENS;
    for (int i = 0; i < N_COMBS; i++)
    {
        // Scale maximum lengths by the scale factor (fs/44100 * max_room)
        comb_lengths[i] *= rv_scale_fac;
        rv.combs[i] = comb_fv_init(
            comb_lengths[i],
            comb_lengths[i] * room_size, // < max_delay
            feedback_int,
            damping_int,
            &memory_manager);
    }
    for (int i = 0; i < N_APS; i++)
    {
        // Scale maximum lengths by the scale factor (fs/44100 * max_room)
        ap_lengths[i] *= rv_scale_fac;
        rv.allpasses[i] = allpass_fv_init(
            ap_lengths[i],
            ap_lengths[i] * room_size,
            Q(Q_RV)(DEFAULT_AP_FEEDBACK),
            &memory_manager);
    }
    return rv;
}

void adsp_reverb_room_reset_state(reverb_room_t *rv)
{
    for (int comb = 0; comb < N_COMBS; comb++)
    {
        comb_fv_reset_state(&rv->combs[comb]);
    }
    for (int ap = 0; ap < N_APS; ap++)
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
    int32_t ah = 0, al = 0, zero = 0, shift = 30;
    // For larger rooms, increase max_room_size
    xassert(new_room_size > 0 && new_room_size <= 1);
    int32_t room_size_int = Q30(new_room_size);
    rv->room_size = room_size_int;
    for (int comb = 0; comb < N_COMBS; comb++)
    {
        // Do chan.comb_lengths[comb] * new_room_size in Q30
        int32_t l = rv->combs[comb].max_delay;
        asm volatile("lmul %0, %1, %2, %3, %4, %5"
                     : "=r"(ah), "=r"(al)
                     : "r"(room_size_int), "r"(l), "r"(zero), "r"(zero));
        asm volatile("lextract %0, %1, %2, %3, 32"
                     : "=r"(ah)
                     : "r"(ah), "r"(al), "r"(shift));
        comb_fv_set_delay(&rv->combs[comb], ah);
    }
    for (int ap = 0; ap < N_APS; ap++)
    {
        // Do chan.ap_lengths[ap] * new_room_size in Q30
        int32_t l = rv->allpasses[ap].max_delay;
        asm volatile("lmul %0, %1, %2, %3, %4, %5"
                     : "=r"(ah), "=r"(al)
                     : "r"(room_size_int), "r"(l), "r"(zero), "r"(zero));
        asm volatile("lextract %0, %1, %2, %3, 32"
                     : "=r"(ah)
                     : "r"(ah), "r"(al), "r"(shift));
        allpass_fv_set_delay(&rv->allpasses[ap], ah);
    }
}

int32_t adsp_reverb_room(
    reverb_room_t *rv,
    int32_t new_samp)
{
    int32_t reverb_input = apply_gain_q31(new_samp, rv->pre_gain);
    int32_t output = 0;
    int64_t acc = 0;
    for (int comb = 0; comb < N_COMBS; comb++)
    {
        acc += comb_fv(&(rv->combs[comb]), reverb_input);
    }
    output = adsp_saturate_32b(acc);

    for (int ap = 0; ap < N_APS; ap++)
    {
        acc = allpass_fv(&(rv->allpasses[ap]), output);
        output = (int32_t)acc; // We do not saturate here!
    }
    acc = apply_gain_q31(output, rv->wet_gain);
    acc += apply_gain_q31(new_samp, rv->dry_gain);
    output = adsp_saturate_32b(acc);

    return output;
}
