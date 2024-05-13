// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include <xcore/chanend.h>
#include <xcore/channel.h>
#include <xcore/parallel.h>
#include <stddef.h>
#include "stages/adsp_module.h"

DECLARE_JOB(dsp_control_thread, (chanend_t, module_instance_t **, size_t));
DECLARE_JOB(dsp_thread, (chanend_t, chanend_t, module_instance_t**, size_t));
