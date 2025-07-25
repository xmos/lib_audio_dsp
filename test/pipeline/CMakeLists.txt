
cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(pipeline_test)

set(APP_HW_TARGET XCORE-AI-EXPLORER)

set(COMMON_APP_COMPILER_FLAGS
    -O3
    -g
    -report
    -Wall
    -Werror
    -fxscope
    -DTEST_WAV_XSCOPE=1
)
set(APP_COMPILER_FLAGS_default
    ${COMMON_APP_COMPILER_FLAGS}
)

set(APP_COMPILER_FLAGS_tuned
    ${COMMON_APP_COMPILER_FLAGS}
    -DSEND_TEST_CONTROL_COMMANDS=1
)

set(APP_DEPENDENT_MODULES lib_audio_dsp "xscope_fileio(v1.3.1)")

set(APP_INCLUDES app_pipeline app_pipeline/file_utils ../../lib_audio_dsp/api/stages/ build/dsp_pipeline_default/ build/ build/app_pipeline.autogen/host/ .)

file(GLOB C_SRC CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_LIST_DIR} app_pipeline/*.c app_pipeline/*.xc app_pipeline/file_utils/*.c stages/*.c)

list(APPEND APP_C_SRCS
    ${C_SRC}
)

# Handle the files specific to each build config.
if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/build/dsp_pipeline_default/adsp_generated_auto.c)
    set(SOURCE_FILES_default build/dsp_pipeline_default/adsp_generated_auto.c)
    list(APPEND APP_C_SRCS build/dsp_pipeline_default/adsp_generated_auto.c)
endif()
if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/build/dsp_pipeline_tuned/adsp_generated_auto.c)
    set(SOURCE_FILES_tuned build/dsp_pipeline_tuned/adsp_generated_auto.c)
    list(APPEND APP_C_SRCS build/dsp_pipeline_tuned/adsp_generated_auto.c)
endif()


file(GLOB ADSP_ADDITIONAL_STAGE_CONFIG  ${CMAKE_CURRENT_LIST_DIR}/stages/*.yaml )
set(XMOS_SANDBOX_DIR ${CMAKE_SOURCE_DIR}/../../..)

XMOS_REGISTER_APP()
