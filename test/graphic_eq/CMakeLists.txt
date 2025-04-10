cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(graphic_eq_test)

set(APP_HW_TARGET XCORE-AI-EXPLORER)

set(APP_COMPILER_FLAGS
    -O3
    -g
    -report
    -Wall
    -Werror
    -fxscope)

set(APP_DEPENDENT_MODULES lib_audio_dsp)
set(APP_C_SRCS src/main.c)

set(XMOS_SANDBOX_DIR ${CMAKE_SOURCE_DIR}/../../..)

XMOS_REGISTER_APP()

project(graphic_eq_coeff_test)

set(APP_HW_TARGET XCORE-AI-EXPLORER)

set(APP_COMPILER_FLAGS
    -O3
    -g
    -report
    -Wall
    -Werror
    -fxscope
    -fcmdline-buffer-bytes=1024)

set(APP_DEPENDENT_MODULES lib_audio_dsp)
set(APP_C_SRCS src/main_coeffs.c)

set(XMOS_SANDBOX_DIR ${CMAKE_SOURCE_DIR}/../../..)

XMOS_REGISTER_APP()
