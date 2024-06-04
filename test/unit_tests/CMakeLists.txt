
cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(unit_test)

set(APP_HW_TARGET XCORE-AI-EXPLORER)
set(XMOS_SANDBOX_DIR ${CMAKE_SOURCE_DIR}/../../..)
set(APP_DEPENDENT_MODULES lib_unity lib_audio_dsp)

set(APP_INCLUDES build/dummy_pipeline ${CMAKE_CURRENT_LIST_DIR})
file(GLOB APP_C_SRCS CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_LIST_DIR} src/*.c build/dummy_pipeline/*.c)
file(GLOB ADSP_ADDITIONAL_STAGE_CONFIG  ${CMAKE_CURRENT_LIST_DIR}/stages/*.yaml )

file(GLOB tests RELATIVE ${CMAKE_CURRENT_LIST_DIR} CONFIGURE_DEPENDS src/test*.c)
foreach(test_file ${tests})
    get_filename_component(test_name ${test_file} NAME_WE)
    set(SOURCE_FILES_${test_name} ${test_file})
    set(APP_COMPILER_FLAGS_${test_name}
        -O0
        -g
        -report
        -Wall
        # -Werror
        -fxscope
        -DDEBUG_PRINT_ENABLE=1)
endforeach()

set(LIB_AUDIO_DSP_DISABLE_OPTIMISATION ON)
set(LIB_UNITY_AUTO_TEST_RUNNER ON)
set(LIB_UNITY_USE_FIXTURE OFF)
set(LIB_UNITY_USE_MEMORY OFF)
XMOS_REGISTER_APP()
