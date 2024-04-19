// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "xmath/xmath.h"

// Default signal exponent
#define SIG_EXP (-27)
// Default Q format
#define Q_SIG   (-SIG_EXP)

#include "dsp/signal_chain.h"
#include "dsp/biquad.h"
#include "dsp/cascaded_biquads.h"
#include "dsp/drc.h"
#include "dsp/reverb.h"
