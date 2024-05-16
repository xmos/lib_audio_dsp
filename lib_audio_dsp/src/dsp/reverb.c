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
#define TWO_TO_31 2147483648
#define TWO_TO_31_MINUS_1 2147483647

#define Q_RVR 31
#define DEFAULT_AP_FEEDBACK 0x40000000 // 0.5 in Q31

static inline int32_t float_to_Q_RVR_pos(float val)
{
    // only works for positive values
    xassert(val >= 0);
    int32_t sign, exp, mant;
    asm("fsexp %0, %1, %2": "=r"(sign), "=r"(exp): "r"(val));
    asm("fmant %0, %1": "=r"(mant): "r"(val));
    // mant to q_rvr
    right_shift_t shr = -Q_RVR - exp + 23;
    mant >>= shr;
    return mant;
}

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

int32_t adsp_reverb_room_calc_gain(float gain_db)
{
    xassert(gain_db > ADSP_RVR_MIN_GAIN_DB &&
            gain_db <= ADSP_RVR_MAX_GAIN_DB);
    int32_t gain = float_to_Q_RVR_pos(DBTOGAIN(gain_db));
    return gain;
}

void adsp_reverb_room_init_filters(
    reverb_room_t *rv,
    float fs,
    float max_room_size,
    int32_t feedback,
    int32_t damping,
    void * reverb_heap)
{
    mem_manager_t memory_manager = mem_manager_init(
        reverb_heap,
        ADSP_RVR_HEAP_SZ(fs, max_room_size));

    // Scale the wet gain; when pregain changes, overall wet gain shouldn't
    const float rv_scale_fac = ADSP_RVR_SCALE(fs, max_room_size);

    // shift 2 insted of / 4, to avoid division
    rv->total_buffer_length = ADSP_RVR_HEAP_SZ(fs, max_room_size) >> 2;

    uint32_t comb_lengths[8] = DEFAULT_COMB_LENS;
    uint32_t ap_lengths[4] = DEFAULT_AP_LENS;
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
    float wet_gain,
    float dry_gain,
    float pregain,
    void *reverb_heap)
{
    // For larger rooms, increase max_room_size. Don't forget to also increase
    // the size of reverb_heap
    xassert(room_size >= 0 && room_size <= 1);
    xassert(decay >= 0 && decay <= 1);
    xassert(damping >= 0 && damping <= 1);

    // These limits should be reconsidered if Q_RVR != 31
    xassert(pregain >= 0 && pregain < 1);

    reverb_room_t rv;

    // Avoids too much or too little feedback
    const int32_t feedback_int = float_to_Q_RVR_pos((decay * 0.28) + 0.7);
    const int32_t damping_int = MAX(float_to_Q_RVR_pos(damping) - 1, 1);

    adsp_reverb_room_init_filters(&rv, fs, max_room_size, feedback_int, damping_int, reverb_heap);
    adsp_reverb_room_set_room_size(&rv, room_size);

    rv.pre_gain = float_to_Q_RVR_pos(pregain);
    rv.dry_gain = adsp_reverb_room_calc_gain(dry_gain);
    rv.wet_gain = adsp_reverb_room_calc_gain(wet_gain);

    return rv;
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
    // could use uq32 for the room size, but it's important
    // to represent 1.0 here, so loosing one bit of precision
    // and doing extra lextracts :(
    int32_t zero = 0, q = 31, exp;
    uint32_t ah = 0, al = 0, room_size_int;
    asm("fsexp %0, %1, %2": "=r"(zero), "=r"(exp): "r"(new_room_size));
    asm("fmant %0, %1": "=r"(room_size_int): "r"(new_room_size));
    right_shift_t shr = -q - exp + 23;
    room_size_int >>= shr;

    rv->room_size = room_size_int;
    for (int comb = 0; comb < ADSP_RVR_N_COMBS; comb++)
    {
        // Do comb length * new_room_size in UQ31
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
        // Do ap length * new_room_size in UQ31
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
    int32_t reverb_input = apply_gain_q31(new_samp, rv->pre_gain);
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
    acc = apply_gain_q31(output, rv->wet_gain);
    acc += apply_gain_q31(new_samp, rv->dry_gain);
    output = adsp_saturate_32b(acc);

    return output;
}
