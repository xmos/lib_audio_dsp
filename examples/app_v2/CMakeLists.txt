cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(app_v2)

set(APP_DEPENDENT_MODULES
    "lib_audio_dsp"
)

set(APP_COMPILER_FLAGS
    -g
    -mcmodel=large
    # -O3
    -report)

set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)
set(APP_HW_TARGET XK-EVK-XU316)
XMOS_REGISTER_APP()
