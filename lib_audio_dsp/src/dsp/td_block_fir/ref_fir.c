// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

// Code for reference: accurate but slow
// prod_shr prevents accumulator overflow
// accu_shr returns the accumulator to the correct output q value
#include "dsp/fir.h"

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
