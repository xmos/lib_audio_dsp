
#pragma once

int32_t adsp_dB_to_gain(float dB_gain);

int32_t adsp_from_q31(int32_t input);

int32_t adsp_to_q31(int32_t input);

int32_t adsp_adder(int32_t * input, unsigned n_ch);

int32_t adsp_subtractor(int32_t x, int32_t y);

int32_t adsp_fixed_gain(int32_t input, int32_t gain);

int32_t adsp_mixer(int32_t * input, unsigned n_ch, int32_t gain);
