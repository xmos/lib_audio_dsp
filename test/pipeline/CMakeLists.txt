
cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(pipeline_test)

set(APP_HW_TARGET XCORE-AI-EXPLORER)

set(APP_COMPILER_FLAGS
    -O3
    -g
    -report
    -Wall
    -Werror
    -fxscope
    -DTEST_WAV_XSCOPE=1)

set(APP_DEPENDENT_MODULES lib_audio_dsp "xscope_fileio(develop)")

set(APP_INCLUDES src src/file_utils build/dsp_pipeline/ .)

file(GLOB C_SRC CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_LIST_DIR} src/*.c src/*.xc src/file_utils/*.c stages/*.c)

set(APP_C_SRCS
    "${C_SRC};build/dsp_pipeline/adsp_generated_auto.c")

file(GLOB ADSP_ADDITIONAL_STAGE_CONFIG  ${CMAKE_CURRENT_LIST_DIR}/stages/*.yaml )
set(XMOS_SANDBOX_DIR ${CMAKE_SOURCE_DIR}/../../..)

XMOS_REGISTER_APP()

project(pipeline_test_stage_control)

set(APP_HW_TARGET XCORE-AI-EXPLORER)

set(APP_COMPILER_FLAGS
    -O3
    -g
    -report
    -Wall
    -Werror
    )

set(APP_DEPENDENT_MODULES lib_audio_dsp)

set(APP_INCLUDES src_test_stage_control build/dsp_pipeline/)

file(GLOB C_SRC CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_LIST_DIR} src_test_stage_control/*.c)

set(APP_C_SRCS
    "${C_SRC};build/dsp_pipeline/adsp_generated_auto.c")

file(GLOB XC_SRC CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_LIST_DIR} src_test_stage_control/*.xc)
set(APP_XC_SRCS "${XC_SRC}")

file(GLOB ADSP_ADDITIONAL_STAGE_CONFIG  ${CMAKE_CURRENT_LIST_DIR}/stages/*.yaml )
set(XMOS_SANDBOX_DIR ${CMAKE_SOURCE_DIR}/../../..)

XMOS_REGISTER_APP()
