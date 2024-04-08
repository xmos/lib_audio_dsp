cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)

project(reverb_test)

set(APP_HW_TARGET XCORE-AI-EXPLORER)

set(APP_COMPILER_FLAGS
    #-O3
    -g
    -report
    -Wall
    -Werror
    -fxscope)

set(APP_DEPENDENT_MODULES lib_audio_dsp)
set(APP_C_SRCS src/reverb_room.c)

set(XMOS_SANDBOX_DIR ${CMAKE_SOURCE_DIR}/../../..)

XMOS_REGISTER_APP()

project(comb_test)

set(APP_HW_TARGET XCORE-AI-EXPLORER)

set(APP_COMPILER_FLAGS
    #-O3
    -g
    -report
    -Wall
    -Werror
    -fxscope)

set(APP_DEPENDENT_MODULES lib_audio_dsp)
set(APP_C_SRCS src/comb_filt.c)

set(XMOS_SANDBOX_DIR ${CMAKE_SOURCE_DIR}/../../..)

XMOS_REGISTER_APP()

project(allpass_test)

set(APP_HW_TARGET XCORE-AI-EXPLORER)

set(APP_COMPILER_FLAGS
    #-O3
    -g
    -report
    -Wall
    -Werror
    -fxscope)

set(APP_DEPENDENT_MODULES lib_audio_dsp)
set(APP_C_SRCS src/allpass_filt.c)

set(XMOS_SANDBOX_DIR ${CMAKE_SOURCE_DIR}/../../..)

XMOS_REGISTER_APP()
