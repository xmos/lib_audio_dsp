
cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(pipeline_test)

set(APP_HW_TARGET XCORE-AI-EXPLORER)

set(APP_COMPILER_FLAGS_initialized
    -O3
    -g
    -report
    -Wall
    -Werror
    -fxscope
    -DTEST_WAV_XSCOPE=1)

set(APP_COMPILER_FLAGS_control_commands
    -O3
    -g
    -report
    -Wall
    -Werror
    -fxscope
    -DTEST_WAV_XSCOPE=1
    -DSEND_TEST_CONTROL_COMMANDS=1)

set(APP_DEPENDENT_MODULES lib_audio_dsp "xscope_fileio(develop)")

set(APP_INCLUDES src src/file_utils build/dsp_pipeline_initialized/ build/src.autogen/host/ .)

file(GLOB C_SRC CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_LIST_DIR} src/*.c src/*.xc src/file_utils/*.c stages/*.c)

#set(APP_C_SRCS ${C_SRC})
set(SOURCE_FILES_initialized build/dsp_pipeline_initialized/adsp_generated_auto.c)
set(SOURCE_FILES_control_commands build/dsp_pipeline_uninitialized/adsp_generated_auto.c)

file(GLOB ADSP_ADDITIONAL_STAGE_CONFIG  ${CMAKE_CURRENT_LIST_DIR}/stages/*.yaml )
set(XMOS_SANDBOX_DIR ${CMAKE_SOURCE_DIR}/../../..)

XMOS_REGISTER_APP()
