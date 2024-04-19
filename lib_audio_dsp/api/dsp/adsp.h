// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include "xmath/xmath.h"

<<<<<<< HEAD
#include "dsp/defines.h"
=======
/**  Default signal exponent */
#define SIG_EXP (-27)
/** Default Q format */
#define Q_SIG   (-SIG_EXP)

>>>>>>> 7ac5aea736a535853237820bf7cc5bcd8d88fa8e
#include "dsp/signal_chain.h"
#include "dsp/biquad.h"
#include "dsp/cascaded_biquads.h"
#include "dsp/drc.h"
#include "dsp/reverb.h"
