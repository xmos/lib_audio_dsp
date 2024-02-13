set(LIB_C_SRCS "")

# TODO Let the user specify their own extra stages
#
# The sources in the "stages" subdirectories of src/ and api/ require
# some code generation to take place. The below implements the code
# generation using some python that is a part of this repo.
#
# As this repo should also be available as a general purpose DSP library
# for which no code generation is requried it is desired that installing the
# python dependencies should not be required for that use case. Therefore
# the below checks if the dependencies are available. If they are then it
# always adds the code gen to the build and it is up to Make to decide
# if the are needed. If the dependencies are not present, then the auto gen
# will not be added to the build, a message is printed, and any build which
# uses the stages api will fail at compile time.
set(STAGES_INCLUDED OFF)
find_program(PYTHON_EXE python NO_CACHE)
if(PYTHON_EXE)
    execute_process(COMMAND ${PYTHON_EXE} -c "import audio_dsp"
                    OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE AUDIO_DSP_NOT_INSTALLED)
    if(NOT ${AUDIO_DSP_NOT_INSTALLED})

        set(ADSP_ADDITIONAL_STAGE_CONFIG "" CACHE STRING "semicolon separated list of stage yaml config files")

        set(STAGES_INCLUDED ON)
        set(AUTOGEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/src.autogen )
        set(LIB_AUDIO_DSP_PATH ${CMAKE_CURRENT_LIST_DIR})
        set(CONFIG_YAML_PATH ${LIB_AUDIO_DSP_PATH}/../stage_config)
        file(GLOB MODULE_CONFIG_YAML_FILES  ${CONFIG_YAML_PATH}/*.yaml )
        list(APPEND MODULE_CONFIG_YAML_FILES ${ADSP_ADDITIONAL_STAGE_CONFIG})
        file(GLOB TEMPLATE_FILES ${LIB_AUDIO_DSP_PATH}/../python/audio_dsp/design/templates/*.mako)
        set(ALL_CONFIG_YAML_DIR ${AUTOGEN_DIR}/yaml)
        unset(CMD_MAP_GEN_ARGS)
        list(APPEND CMD_MAP_GEN_ARGS --config-dir ${ALL_CONFIG_YAML_DIR} --out-dir ${AUTOGEN_DIR})
        set(CMD_MAP_GEN_SCRIPT ${LIB_AUDIO_DSP_PATH}/../python/audio_dsp/design/parse_config.py)

        # Get output C file names
        set(OUTPUT_C_FILES ${AUTOGEN_DIR}/generator/gen_cmd_map_offset.c)

        # output h file names
        set(COPIED_YAML_FILES "")
        set(OUTPUT_H_FILES ${AUTOGEN_DIR}/common/cmds.h ${AUTOGEN_DIR}/device/cmd_offsets.h ${AUTOGEN_DIR}/host/host_cmd_map.h)
        foreach(YAML_FILE ${MODULE_CONFIG_YAML_FILES})
            get_filename_component(STAGE_NAME ${YAML_FILE} NAME_WE)
            list(APPEND OUTPUT_H_FILES ${AUTOGEN_DIR}/common/${STAGE_NAME}_config.h)

            # copy all yaml files to the same directory so 
            # they can be used by generation script
            set(copied_config ${ALL_CONFIG_YAML_DIR}/${STAGE_NAME}.yaml)
            add_custom_command(
                OUTPUT ${copied_config}
                COMMAND ${CMAKE_COMMAND} -E copy ${YAML_FILE} ${copied_config}
                DEPENDS ${YAML_FILE}
                COMMENT "Copying ${STAGE_NAME}.yaml"
                VERBATIM
            )
            list(APPEND COPIED_YAML_FILES ${copied_config}) 
        endforeach()

        add_custom_command(
            OUTPUT ${OUTPUT_C_FILES} ${OUTPUT_H_FILES}
            COMMAND ${PYTHON_EXE} -m audio_dsp.design.parse_config ${CMD_MAP_GEN_ARGS}
            DEPENDS ${COPIED_YAML_FILES} ${CMD_MAP_GEN_SCRIPT} ${TEMPLATE_FILES}
            COMMENT "Generating cmd_map files included in the device and host application"
            VERBATIM
        )

        file(RELATIVE_PATH REL_AUTOGEN_DIR ${CMAKE_CURRENT_LIST_DIR} ${AUTOGEN_DIR})
        set(PIPELINE_DESIGN_INCLUDE_DIRS ${REL_AUTOGEN_DIR}/common ${REL_AUTOGEN_DIR}/device)

        if(NOT TARGET cmd_map_generation)
            add_custom_target(cmd_map_generation
                DEPENDS ${OUTPUT_C_FILES} ${OUTPUT_H_FILES})
        endif()
        
        file(GLOB STAGES_C_SOURCES RELATIVE ${CMAKE_CURRENT_LIST_DIR} CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/stages/*.c")
        list(APPEND LIB_C_SRCS ${STAGES_C_SOURCES})
    else()
        message("Excluding lib_audio_dsp stages as audio_dsp python package not available")
    endif()
else()
    message("Excluding lib_audio_dsp stages as python not available")
endif()


set(LIB_NAME lib_audio_dsp)
set(LIB_VERSION 0.1.0)
set(LIB_INCLUDES api ${PIPELINE_DESIGN_INCLUDE_DIRS})
file(GLOB DSP_C_SOURCES RELATIVE ${CMAKE_CURRENT_LIST_DIR} CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/dsp/*.c")
list(APPEND LIB_C_SRCS ${DSP_C_SOURCES})
set(LIB_DEPENDENT_MODULES "lib_xcore_math(xcommon_cmake)" "lib_logging")
set(LIB_COMPILER_FLAGS -O3 -Wall -Werror -g )
set(LIB_OPTIONAL_HEADERS adsp_generated_auto.h)

XMOS_REGISTER_MODULE()

if(STAGES_INCLUDED)
    foreach(target ${APP_BUILD_TARGETS})
        add_dependencies(${target} cmd_map_generation)
    endforeach()
endif()
