// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"
#include <xcore/assert.h>


int32_t adsp_graphic_eq_10b(int32_t new_sample,
                                int32_t gains[10],
                                q2_30 coeffs[50],
                                int32_t state[160])
{
    int32_t ah = 0, al = 1 << (Q_GEQ - 1);
    int state_idx = 0;
    int32_t polarity = -1;

    #pragma unroll
    for(unsigned n = 0; n < 10; n++)
    {
        int32_t this_band;
        
        this_band = adsp_biquad(new_sample, &coeffs[5 * n], &state[state_idx], 0);
        state_idx += 8;
        this_band = adsp_biquad(this_band, &coeffs[5 * n], &state[state_idx], 0);
        state_idx += 8;

        // alternate summing and subtracting bands
        polarity *= -1;
        int32_t this_gain = gains[n] * polarity;
        asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (this_band), "r" (this_gain), "0" (ah), "1" (al));
    }

    asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (Q_GEQ), "0" (ah), "1" (al));
    asm("lextract %0, %1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (Q_GEQ));

    return ah;
}
