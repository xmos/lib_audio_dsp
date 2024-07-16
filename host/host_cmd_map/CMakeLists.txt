## Raspbian version that we currently use 3.13  is the latest possible version of cmake
## This cmake file will not be integrated with the top-level cmake structure till this is resolved
cmake_minimum_required(VERSION 3.13)
# Force release mode so that it won't use debug dlls
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Only Release mode is allowed" FORCE)
project(xvf3800_host_app)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)


set(RELEASE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../build/release)

if((UNIX AND NOT APPLE) AND (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv7l"))
    set(CMAKE_INSTALL_PREFIX ${RELEASE_DIR}/host/rpi)
elseif((UNIX AND NOT APPLE) AND (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64"))
    set(CMAKE_INSTALL_PREFIX ${RELEASE_DIR}/host/linux_x86_64)
elseif(APPLE AND (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64"))
    set(CMAKE_INSTALL_PREFIX ${RELEASE_DIR}/host/mac_x86_64)
elseif(APPLE AND (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64"))
    set(CMAKE_INSTALL_PREFIX ${RELEASE_DIR}/host/mac_arm64)
elseif(WIN32)
    set(CMAKE_INSTALL_PREFIX ${RELEASE_DIR}/host/win32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    include(GenerateExportHeader)
else()
    message(FATAL_ERROR "The platform you are using is not supported")
endif()

enable_language(CXX C)

## Define YAML autogen macro
#*********************************************************
# Autogenerate files from the cmds .yaml file
#*********************************************************
set(LIB_AUDIO_DSP_PATH ${CMAKE_CURRENT_LIST_DIR}/../../lib_audio_dsp)
set(CONFIG_YAML_PATH ${LIB_AUDIO_DSP_PATH}/../stage_config)
set( AUTOGEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/src.autogen )
file( GLOB CONFIGRE_DEPENDS MODULE_CONFIG_YAML_FILES  ${CONFIG_YAML_PATH}/*.yaml )
file(GLOB TEMPLATE_FILES ${LIB_AUDIO_DSP_PATH}/../python/audio_dsp/design/templates/*.mako)
unset(CMD_MAP_GEN_ARGS)
list(APPEND CMD_MAP_GEN_ARGS --config-dir ${CONFIG_YAML_PATH} --out-dir ${AUTOGEN_DIR})
set(CMD_MAP_GEN_SCRIPT ${LIB_AUDIO_DSP_PATH}/../python/audio_dsp/design/parse_config.py)

# Get output C file names
set(OUTPUT_C_FILES ${AUTOGEN_DIR}/generator/gen_cmd_map_offset.c)

add_custom_command(
    OUTPUT ${OUTPUT_C_FILES}
    COMMAND python -m audio_dsp.design.parse_config ${CMD_MAP_GEN_ARGS}
    DEPENDS ${MODULE_CONFIG_YAML_FILES} ${CMD_MAP_GEN_SCRIPT} ${TEMPLATE_FILES}
    COMMENT "Generating cmd_map files included in the device and host application"
    VERBATIM
)

add_custom_target(cmd_map_generation
    DEPENDS ${OUTPUT_C_FILES})


# Add target to generate the NUM_VALUES defines file
set(APP_HW_TARGET XK-EVK-XU316)
set(NUM_VALUES_GEN_APP gen_cmd_num_values)
add_executable(${NUM_VALUES_GEN_APP})
add_dependencies(${NUM_VALUES_GEN_APP} cmd_map_generation)
target_sources(${NUM_VALUES_GEN_APP} PRIVATE ${OUTPUT_C_FILES})
target_include_directories(${NUM_VALUES_GEN_APP} PRIVATE
                ${AUTOGEN_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../../lib_xcore_math/lib_xcore_math/api
                ${LIB_AUDIO_DSP_PATH}/api ${AUTOGEN_DIR}/common ${AUTOGEN_DIR}/device ${AUTOGEN_DIR}/generator)
target_compile_options(${NUM_VALUES_GEN_APP} PRIVATE -DOFFSET_GEN=1 -g)


add_custom_target(run_${NUM_VALUES_GEN_APP}
COMMAND $<TARGET_FILE:${NUM_VALUES_GEN_APP}> ${AUTOGEN_DIR}/host/cmd_num_values.h
DEPENDS ${NUM_VALUES_GEN_APP}
COMMENT
  "Generate cmd num values defines included in the host code"
VERBATIM
)

#*************************************
# Add the command map shared library
#*************************************
add_library(command_map SHARED ${CMAKE_CURRENT_LIST_DIR}/command_map.cpp)
add_dependencies(command_map run_${NUM_VALUES_GEN_APP})
target_link_options(command_map PRIVATE -fPIC)
target_include_directories(command_map PUBLIC ${DEVICE_CONTROL_PATH}/api)
target_include_directories(command_map PUBLIC ${AUTOGEN_DIR}/common ${AUTOGEN_DIR}/host)
target_sources(command_map PRIVATE ${CMAKE_CURRENT_LIST_DIR}/command_map.cpp)
target_compile_definitions(command_map
    PRIVATE
        _USE_MATH_DEFINES=1
)

if(WIN32)
    generate_export_header(command_map)
endif()

add_custom_command(TARGET command_map
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:command_map> ${CMAKE_INSTALL_PREFIX}/$<TARGET_FILE_NAME:command_map>
)

message(STATUS "Building with ${CMAKE_C_COMPILER_ID}")
