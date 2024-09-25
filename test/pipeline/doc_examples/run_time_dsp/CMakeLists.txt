cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(run_time_dsp)

set(APP_HW_TARGET XK-EVK-XU316)
set(APP_DEPENDENT_MODULES lib_audio_dsp)
set(APP_PCA_ENABLE OFF)
set(APP_COMPILER_FLAGS -Wall -Werror)
set(APP_INCLUDES
    src
    src/dsp)
set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../../..)

XMOS_REGISTER_APP()
