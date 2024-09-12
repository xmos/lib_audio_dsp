cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(app_simple_audio_dsp_integration)

set(APP_HW_TARGET XK-EVK-XU316)
set(APP_DEPENDENT_MODULES
    "lib_audio_dsp"
    "lib_logging(3.2.0)"
    "lib_locks(2.2.0)"
)
set(APP_PCA_ENABLE OFF)
set(EXAMPLE_BUILD_FLAGS ${EXTRA_BUILD_FLAGS} -fcomment-asm
                                             -Wall
                                             -Os
                                             -report
                                             -lquadflash
                                             -mcmodel=large
                                             -g
                                             -fxscope)

set(APP_COMPILER_FLAGS ${EXAMPLE_BUILD_FLAGS})

file(GLOB C_SRC CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_LIST_DIR} src/*.c)

set(DSP_DIR build/dsp_pipeline)
# set(DSP_MAIN ${DSP_DIR}/adsp_generated_auto.c)
# if(NOT EXISTS ${DSP_MAIN})
#     write_file(${DSP_MAIN} "#error Generate pipeline before building or build with dsp_pipeline.ipynb")
#     write_file(${DSP_DIR}/adsp_generated_auto.h "#error Generate pipeline before building or build with dsp_pipeline.ipynb")
# endif()

set(APP_C_SRCS
    "${C_SRC};${DSP_MAIN}")


set(APP_INCLUDES
    src
    src/core
    src/extensions
    ${CMAKE_CURRENT_LIST_DIR}/build/dsp_pipeline)
set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)

XMOS_REGISTER_APP()
