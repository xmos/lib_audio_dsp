// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"
#include <xcore/assert.h>
#include "control/biquad.h"

//hand tuned values 16k (8 band)
static const float cfs_16[8] = {31.125, 64, 125, 250, 500, 1000, 2000, 4200};
static const float bw_16[8] = {1.5175, 1.6175, 1.5175, 1.5175, 1.5175, 1.5175, 1.6175, 1.1};
static const float gains_16[8] = {-0.3, -0.225, 0.175, 0, 0.05, 0.15, -0.4, -0.175};

//hand tuned values 32k (9 band)
static const float cfs_32[9] = {31.125, 64, 125, 250, 500, 1000, 2000, 4000, 8500};
static const float bw_32[9] = {1.5175, 1.6175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 1.1};
static const float gains_32[9] = {-0.3, -0.225, 0.175, 0, 0.01, 0, 0.075, -0.35, -0.1};

// hand tuned values for 44.1k and 48k
static const float cfs_46[10] = {31.125, 64, 125, 250, 500, 1000, 2000, 4000, 8150, 15000};
static const float bw_46[10] = {1.5175, 1.6175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 0.75875};
static const float gains_46[10] = {-0.3, -0.225, 0.1750, 0, 0.01, 0, 0.05, 0.025, -0.41, -0.25};

// hand tuned values for 88.2k and 96k
static const float  cfs_92[10] = {31.125, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000};
static const float  bw_92[10] = {1.5175, 1.6175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 1.1};
static const float  gains_92[10] = {-0.3, -0.225, 0.1750, 0, 0.01, 0, 0.0, 0.0, -0.35, -0.2};

// hand tuned values for 192k
static const float cfs_192[10] = {31.125, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000};
static const float bw_192[10] = {1.5175, 1.6175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 1.5175, 1.1};
static const float gains_192[10] = {-0.3, -0.225, 0.1750, 0, 0.01, 0, 0.0, -0.05, -0.3, -0.2};

q2_30* adsp_graphic_eq_10b_init(float fs)
{

    static q2_30 coeffs[50] = {0};
    float shift = -0.37;

    xassert(fs > 12000.0f && "10 band Graphic EQ only supports fs > 12k");

    if ( fs <= 16000) {
        int j = 0;
        for(int i = 0; i < 8; i++)
        {
            adsp_design_biquad_bandpass(&coeffs[j], cfs_16[i], fs, bw_16[i]);
            adsp_apply_biquad_gain(&coeffs[j], 0, gains_16[i] + shift);
            j += 5;
        }
    }
    else if (fs <= 32000){
        int j = 0;
        for(int i = 0; i < 9; i++)
        {
            adsp_design_biquad_bandpass(&coeffs[j], cfs_32[i], fs, bw_32[i]);
            adsp_apply_biquad_gain(&coeffs[j], 0, gains_32[i] + shift);
            j += 5;
        }
    }
    else if ( fs < ((48000.0f + 88200.0f) / 2.0f)) {
        int j = 0;
        for(int i = 0; i < 10; i++)
        {
            adsp_design_biquad_bandpass(&coeffs[j], cfs_46[i], fs, bw_46[i]);
            adsp_apply_biquad_gain(&coeffs[j], 0, gains_46[i] + shift);
            j += 5;
        }
    }
    else if (fs < (96000 + 176400) / 2){
        int j = 0;
        for(int i = 0; i < 10; i++)
        {
            adsp_design_biquad_bandpass(&coeffs[j], cfs_92[i], fs, bw_92[i]);
            adsp_apply_biquad_gain(&coeffs[j], 0, gains_92[i] + shift);
            j += 5;
        }    
    }
    else {
        int j = 0;
        for(int i = 0; i < 10; i++)
        {
            adsp_design_biquad_bandpass(&coeffs[j], cfs_192[i], fs, bw_192[i]);
            adsp_apply_biquad_gain(&coeffs[j], 0, gains_192[i] + shift);
            j += 5;
        }    
    }

    return coeffs;
}