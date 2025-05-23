cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)

set(CMAKE_OBJECT_PATH_MAX 128)
set(CMAKE_OBJECT_NAME_MAX 1)

project(reverb_test)

set(APP_HW_TARGET XCORE-AI-EXPLORER)

set(APP_COMPILER_FLAGS
    -O3
    -g
    -report
    -Wall
    -Werror
    -fxscope)

set(APP_DEPENDENT_MODULES lib_audio_dsp)
set(APP_C_SRCS src/reverb_room.c)

set(XMOS_SANDBOX_DIR ${CMAKE_SOURCE_DIR}/../../..)

XMOS_REGISTER_APP()


project(reverb_st_test)

set(APP_HW_TARGET XCORE-AI-EXPLORER)

set(APP_COMPILER_FLAGS
    -O3
    -g
    -report
    -Wall
    -Werror
    -fxscope)

set(APP_DEPENDENT_MODULES lib_audio_dsp)
set(APP_C_SRCS src/reverb_room_st.c)

set(XMOS_SANDBOX_DIR ${CMAKE_SOURCE_DIR}/../../..)

XMOS_REGISTER_APP()


project(reverb_plate_test)

set(APP_HW_TARGET XCORE-AI-EXPLORER)

set(APP_COMPILER_FLAGS
    -O3
    -g
    -report
    -Wall
    -Werror
    -fxscope)

set(APP_DEPENDENT_MODULES lib_audio_dsp)
set(APP_C_SRCS src/reverb_plate.c)

set(XMOS_SANDBOX_DIR ${CMAKE_SOURCE_DIR}/../../..)

XMOS_REGISTER_APP()


project(reverb_converters)
set(APP_C_SRCS src/converters.c)
set(APP_COMPILER_FLAGS_FLOAT2INT ${APP_COMPILER_FLAGS} -DFLOAT2INT)
set(APP_COMPILER_FLAGS_DB2INT ${APP_COMPILER_FLAGS} -DDB2INT)
set(APP_COMPILER_FLAGS_DECAY2FEEDBACK ${APP_COMPILER_FLAGS} -DDECAY2FEEDBACK)
set(APP_COMPILER_FLAGS_CALCULATE_DAMPING ${APP_COMPILER_FLAGS} -DCALCULATE_DAMPING)
set(APP_COMPILER_FLAGS_WET_DRY_MIX ${APP_COMPILER_FLAGS} -DWET_DRY_MIX)
set(APP_COMPILER_FLAGS_WET_DRY_MIX_ST ${APP_COMPILER_FLAGS} -DWET_DRY_MIX_ST)
set(APP_COMPILER_FLAGS_CUTOFF ${APP_COMPILER_FLAGS} -DCUTOFF)

unset(APP_COMPILER_FLAGS)
XMOS_REGISTER_APP()
