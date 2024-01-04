cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(biquad_test)


set( AUTOGEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/src.autogen )
set(LIB_AUDIO_DSP_PATH ${CMAKE_CURRENT_LIST_DIR}/../../lib_audio_dsp)
set(CONFIG_YAML_PATH ${LIB_AUDIO_DSP_PATH}/src/dummy_stages/config)
file( GLOB MODULE_CONFIG_YAML_FILES  ${CONFIG_YAML_PATH}/*.yaml )
file(GLOB TEMPLATE_FILES ${LIB_AUDIO_DSP_PATH}/src/dummy_stages/autogen/templates/*.mako)
unset(CMD_MAP_GEN_ARGS)
list(APPEND CMD_MAP_GEN_ARGS --config-dir ${CONFIG_YAML_PATH} --out-dir ${AUTOGEN_DIR})
set(CMD_MAP_GEN_SCRIPT ${LIB_AUDIO_DSP_PATH}/src/dummy_stages/autogen/parse_config.py)

# Get output C file names
set(OUTPUT_C_FILES ${AUTOGEN_DIR}/generator/gen_cmd_map_offset.c)

add_custom_command(
    OUTPUT ${OUTPUT_C_FILES}
    COMMAND python ${CMD_MAP_GEN_SCRIPT} ${CMD_MAP_GEN_ARGS}
    DEPENDS ${MODULE_CONFIG_YAML_FILES} ${CMD_MAP_GEN_SCRIPT} ${TEMPLATE_FILES}
    COMMENT "Generating cmd_map files included in the device and host application"
    VERBATIM
)

add_custom_target(cmd_map_generation
    DEPENDS ${OUTPUT_C_FILES})


set(APP_HW_TARGET XCORE-AI-EXPLORER)
set(APP_INCLUDES src ${AUTOGEN_DIR}/common ${AUTOGEN_DIR}/device)

set(APP_COMPILER_FLAGS
    -Os
    -g
    -report
    -Wall
    -Werror
    -fxscope
    -mcmodel=large)

set(APP_DEPENDENT_MODULES lib_audio_dsp)

set(XMOS_SANDBOX_DIR ${CMAKE_SOURCE_DIR}/../../..)

XMOS_REGISTER_APP()

foreach(BUILD_TARGET ${APP_BUILD_TARGETS})
    add_dependencies(${BUILD_TARGET} cmd_map_generation)
endforeach()
