// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once
#include <xmath/xmath.h>

/**  Default signal exponent */
#define SIG_EXP (-27)
/** Default Q format */
#define Q_SIG   (-SIG_EXP)

typedef struct {
 int32_t exp;
 __attribute__((aligned(8))) complex_s32_t data[];
} complex_spectrum_t;
