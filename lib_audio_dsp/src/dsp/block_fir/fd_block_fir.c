// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <string.h>
#include "xmath/xmath.h"
#include "dsp/fd_block_fir.h"

/**
 * @brief This does a bfp_complex_s32_macc but bin zero is treated as two real values.
 */
__attribute__((noinline)) // bug workaround
static void
bfp_complex_s32_macc2(
    bfp_complex_s32_t *acc,
    const bfp_complex_s32_t *b,
    const bfp_complex_s32_t *c)
{
    exponent_t a_exp;
    right_shift_t acc_shr, b_shr, c_shr;

    vect_complex_s32_macc_prepare(&a_exp, &acc_shr, &b_shr, &c_shr, acc->exp, b->exp, c->exp, acc->hr, b->hr, c->hr);

    acc->exp = a_exp;
    headroom_t hr = vect_s32_macc((int32_t *)acc->data, (int32_t *)b->data, (int32_t *)c->data, 2, acc_shr, b_shr, c_shr);

    acc->hr = vect_complex_s32_macc(acc->data + 1, b->data + 1, c->data + 1, b->length - 1, acc_shr, b_shr, c_shr);

    if (hr < acc->hr)
        acc->hr = hr;
}

static unsigned get_tail(fd_fir_data_t *fir_data)
{
    unsigned tail = fir_data->head_index + 1;
    if (tail == fir_data->block_count)
        tail = 0;
    return tail;
}

static void advance_head(fd_fir_data_t *fir_data)
{
    fir_data->head_index = get_tail(fir_data);
}

/*
This adds fir_data->frame_advance samples to a FIFO of blocks within fir_data.
Blocks wll have overlapping data within them. This also performs a mono FFT.
*/
static void add_data(fd_fir_data_t *fir_data, int32_t *samples_in, exponent_t exp)
{

    // Calc the block to evict, i.e. the oldest one
    unsigned head_tail_idx = get_tail(fir_data);
    int32_t *d = (int32_t *)fir_data->data_blocks[head_tail_idx].data;

    // Copy in the new samples plus the frame overlap
    int32_t prev_len = (fir_data->td_block_length - fir_data->frame_advance);

    memcpy(d, fir_data->prev_td_data, prev_len * sizeof(int32_t));
    memcpy(d + prev_len, samples_in, fir_data->frame_advance * sizeof(int32_t));

    // Update the prev_td_data
    memcpy(fir_data->prev_td_data, d + fir_data->frame_advance,
           prev_len * sizeof(int32_t));

    //[asj] we could optimise the copying to only copy fir_data->frame_advance samples by keeping an index
    // into overlapping frame_data and treating it as a circular buffer

    bfp_s32_t *new_bfp_block = (bfp_s32_t *)&(fir_data->data_blocks[head_tail_idx]);
    bfp_s32_init((bfp_s32_t *)(fir_data->data_blocks + head_tail_idx), d, exp, fir_data->td_block_length, 1);
    bfp_fft_forward_mono(new_bfp_block);
}

#define INTERNAL_EXP (0)  // this is arbitrary
#define ZERO_EXP (-99999) // this is fine

void fd_block_fir_add_data(
    int32_t *samples_in,
    fd_fir_data_t *fir_data)
{
    exponent_t exp = INTERNAL_EXP;
    add_data(fir_data, samples_in, exp);
    advance_head(fir_data);
}

__attribute__((noinline)) // bug workaround
void
fd_block_fir_compute(
    int32_t *samples_out, // must be int32_t samples_out[BLOCK_LENGTH];
    fd_fir_data_t *fir_data,
    fd_fir_filter_t *fir_filter)
{
    assert(fir_data->td_block_length == fir_filter->td_block_length);
    assert(fir_data->block_count >= fir_filter->block_count);

    bfp_complex_s32_t result;
    // data_in does not need clearing as a massively negative exponent takes care of it
    // to make it represent a zero array.
    bfp_complex_s32_init(&result, (complex_s32_t *)samples_out, ZERO_EXP, fir_data->td_block_length / 2, 0);

    memset(samples_out, 0, sizeof(int32_t) * fir_data->td_block_length);

    int idx = fir_data->head_index;
    for (int i = 0; i < fir_filter->block_count; i++)
    {
        bfp_complex_s32_macc2(&result, &(fir_data->data_blocks[idx]), &(fir_filter->coef_blocks[i]));

        if (idx == 0)
            idx = fir_data->block_count - 1;
        else
            idx--;
    }

    bfp_s32_t *time_domain_result = bfp_fft_inverse_mono(&result);

    // denormalise
    exponent_t exp = INTERNAL_EXP;
    bfp_s32_use_exponent(time_domain_result, exp);

    // copy out the result
    int output_samples = fir_filter->td_block_length + 1 - fir_filter->taps_per_block;

    for (int i = 0; i < output_samples; i++)
    {
        samples_out[i] = time_domain_result->data[fir_filter->taps_per_block - 1 + i];
    }
}

void fd_block_fir_data_init(fd_fir_data_t *d, int32_t *data_blob,
                            uint32_t frame_advance, uint32_t td_block_length, uint32_t block_count)
{

    // These are the three properties
    d->td_block_length = td_block_length;
    d->block_count = block_count;
    d->frame_advance = frame_advance;

    uint32_t prev_data_length = d->td_block_length - d->frame_advance;
    bfp_complex_s32_t *bfp_data_blocks = (bfp_complex_s32_t *)data_blob;

    int32_t *data_buffer = (int32_t *)(bfp_data_blocks + d->block_count);

    // if data_buffer isn't double word aligned then force it to be
    if (((int)data_buffer) & 0x7)
        data_buffer += 1;

    for (int i = 0; i < d->block_count; i++)
    {
        bfp_data_blocks[i].data = (complex_s32_t *)data_buffer;
        bfp_data_blocks[i].exp = ZERO_EXP;
        bfp_data_blocks[i].length = d->td_block_length / 2;
        bfp_data_blocks[i].hr = 32;
        bfp_data_blocks[i].flags = 0;
        data_buffer += d->td_block_length; // will always be 8 byte aligned
    }
    d->data_blocks = bfp_data_blocks;
    d->prev_td_data = data_buffer; // will always be 8 byte  aligned
    d->overlapping_frame_data = d->prev_td_data + prev_data_length;
    d->head_index = 0;
}
