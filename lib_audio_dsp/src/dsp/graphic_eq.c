// Copyright 2024-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "dsp/adsp.h"
#include <xcore/assert.h>

q2_30* adsp_graphic_eq_10b_init(float fs)
{
    // coeffs for 44.1k and 48k Hz
    static q2_30 coeffs_46[50] = {2318510, 0, -2318510, 2142455467, -1068732963,
        5142803, 0, -5142803, 2136387310, -1062726942,
        9766005, 0, -9766005, 2127198759, -1053766357,
        18966252, 0, -18966252, 2106674201, -1034158569,
        37290772, 0, -37290772, 2064931417, -996004181,
        71907391, 0, -71907391, 1978846643, -923668491,
        135297944, 0, -135297944, 1797658519, -792990900,
        239288703, 0, -239288703, 1409885624, -575772958,
        378323717, 0, -378323717, 584464918, -246002547,
        363653862, 0, -363653862, -625915388, -292621029,
        };

    // coeffs for 88.2 and 96k Hz
    static q2_30 coeffs_92[50] = {1160608, 0, -1160608, 2144971458, -1071234470,
        2578010, 0, -2578010, 2141941635, -1068220228,
        4905803, 0, -4905803, 2137371560, -1063707454,
        9571212, 0, -9571212, 2127198759, -1053766357,
        18988100, 0, -18988100, 2106674201, -1034158569,
        37247864, 0, -37247864, 2064931417, -996004181,
        71907391, 0, -71907391, 1978846643, -923668491,
        134521344, 0, -134521344, 1797658519, -792990900,
        229177615, 0, -229177615, 1409885624, -575772958,
        303127370, 0, -303127370, 691727862, -426367982
        };

    // coeffs for 192000 Hz
    static q2_30 coeffs_192[50] = {557068, 0, -557068, 2146279057, -1072538347,
        1238296, 0, -1238296, 2144826756, -1071089636,
        2358986, 0, -2358986, 2142640633, -1068916736,
        4613504, 0, -4613504, 2137783552, -1064113274,
        9196431, 0, -9196431, 2128027546, -1054570622,
        18209358, 0, -18209358, 2108350615, -1035738233,
        35787105, 0, -35787105, 2068356649, -999052838,
        68784016, 0, -68784016, 1985963363, -929358338,
        125330682, 0, -125330682, 1812774098, -802980020,
        171037280, 0, -171037280, 1543437362, -708466129,
        };

    xassert(fs > 44000.0f && "10 band Graphic EQ only supports fs > 44k");

    if ( fs < ((48000.0f + 88200.0f) / 2.0f)) {
        return &coeffs_46[0];
    }
    else if (fs < 172000.0f){
        return &coeffs_92[0];
    }
    else {
        return &coeffs_192[0];
    }

}

#define Q_GEQ 29

int32_t adsp_graphic_eq_10b(int32_t new_sample,
                                int32_t gains[10],
                                q2_30 coeffs[50],
                                int32_t state[160])
{
    int32_t ah = 0, al = 1 << (Q_GEQ - 1);
    // int32_t* state_ptr = &state[0];
    int state_idx = 0;

    for(unsigned n = 0; n < 10; n++)
    {
        int32_t this_band;
        // this_band = adsp_biquad(new_sample, &coeffs[5 * n], state_ptr, 0);
        this_band = adsp_biquad(new_sample, &coeffs[5 * n], &state[state_idx], 0);

        // state_ptr += 8*sizeof(int32_t);
        // this_band = adsp_biquad(this_band, &coeffs[5 * n], state_ptr, 0);
        state_idx += 8;
        this_band = adsp_biquad(this_band, &coeffs[5 * n], &state[state_idx], 0);

        int32_t this_gain = n % 2 == 0 ? gains[n] : -gains[n];
        // out += this_band * this_gain;
        asm("maccs %0, %1, %2, %3": "=r" (ah), "=r" (al): "r" (this_band), "r" (this_gain), "0" (ah), "1" (al));

        state_idx += 8;
        // state_ptr += 8*sizeof(int32_t);
    }
    asm("lsats %0, %1, %2": "=r" (ah), "=r" (al): "r" (Q_GEQ), "0" (ah), "1" (al));
    asm("lextract %0, %1, %2, %3, 32": "=r" (ah): "r" (ah), "r" (al), "r" (Q_GEQ));

    return ah;
}
