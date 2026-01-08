// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

// Code for reference: accurate but slow
// prod_shr prevents accumulator overflow
// accu_shr returns the accumulator to the correct output q value
#include "ref_fir.h"
#include <string.h>
#include "xmath/xs3/vpu_scalar_ops.h"


int32_t td_reference_fir(
    int32_t new_sample,
    td_reference_fir_filter_t *filter,
    int32_t *data)
{

    for (uint32_t i = filter->length - 1; i > 0; i--)
        data[i] = data[i - 1];
    data[0] = new_sample;

    int64_t accu = 0;
    for (uint32_t i = 0; i < filter->length; i++)
    {
        int64_t p = (int64_t)data[i] * (int64_t)filter->coefs[i];
        accu += ((p + (1 << (filter->prod_shr - 1))) >> filter->prod_shr);
    }

    int64_t res = (accu + (1 << (filter->accu_shr - 1))) >> filter->accu_shr;
    if (res > INT32_MAX)
        res = INT32_MAX;
    if (res < INT32_MIN)
        res = INT32_MIN;
    return res;
}


void td_block_fir_add_data_ref(
    int32_t samples_in[TD_BLOCK_FIR_LENGTH],
    td_block_fir_data_t *fir_data)
{

    int head;

    // if this is the end of the buffer then paste it onto the front too
    memcpy((void *)fir_data->data + fir_data->index, samples_in, sizeof(int32_t) * TD_BLOCK_FIR_LENGTH);

    if (fir_data->index == fir_data->data_stride)
    {
        memcpy(fir_data->data + 0, samples_in, sizeof(int32_t) * TD_BLOCK_FIR_LENGTH);
        head = 32;
    }
    else
    {
        head = fir_data->index + 32;
    }

    fir_data->index = head;
}


void td_block_fir_compute_ref(
    int32_t output_block[TD_BLOCK_FIR_LENGTH],
    td_block_fir_data_t *fir_data,
    td_block_fir_filter_t *fir_filter)
{

    int64_t accu[TD_BLOCK_FIR_LENGTH];
    memset(accu, 0, sizeof(accu));

    void *data_p = (void *)fir_data->data + fir_data->index + fir_data->data_stride - fir_filter->block_count * 32;

    int second_loop_coutner = (fir_data->index - 32) / 32;
    int first_loop_coutner = fir_filter->block_count - second_loop_coutner;

    if (first_loop_coutner <= 0)
    {
        second_loop_coutner += first_loop_coutner;
        first_loop_coutner = 0;
    }

    void *filter_p = fir_filter->coefs;
    while (first_loop_coutner != 0)
    {
        for (int b = 0; b < TD_BLOCK_FIR_LENGTH; b++)
        {
            accu[TD_BLOCK_FIR_LENGTH - 1 - b] = vlmaccr32(accu[TD_BLOCK_FIR_LENGTH - 1 - b], data_p, filter_p);
            data_p -= 4;
        }
        data_p += 64;
        filter_p += 32;
        first_loop_coutner--;
    }
    data_p -= fir_data->data_stride;
    while (second_loop_coutner != 0)
    {
        for (int b = 0; b < TD_BLOCK_FIR_LENGTH; b++)
        {
            accu[TD_BLOCK_FIR_LENGTH - 1 - b] = vlmaccr32(accu[TD_BLOCK_FIR_LENGTH - 1 - b], data_p, filter_p);
            data_p -= 4;
        }
        data_p += 64;
        filter_p += 32;
        second_loop_coutner--;
    }

    uint32_t accu_shr = fir_filter->accu_shr;
    uint32_t accu_shl = fir_filter->accu_shl;

    for (int i = 0; i < TD_BLOCK_FIR_LENGTH; i++)
    {
        int64_t t = (accu[i] + (1 << (accu_shr - 1))) >> accu_shr;
        int64_t res = t << accu_shl;
        if (res > INT32_MAX)
            res = INT32_MAX;
        if (res < INT32_MIN)
            res = INT32_MIN;
        output_block[i] = res;
    }
}
