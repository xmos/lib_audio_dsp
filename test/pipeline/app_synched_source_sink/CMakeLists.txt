

cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(app_synched_source_sink)

set(APP_HW_TARGET XCORE-AI-EXPLORER)

set(COMMON_APP_COMPILER_FLAGS
    -Os
    -g
    -report
    -Wall
    -Werror
    -fxscope
)

file(RELATIVE_PATH REL_PATH_TO_BUILD "${CMAKE_CURRENT_LIST_DIR}" "${CMAKE_BINARY_DIR}")
set(APP_DEPENDENT_MODULES lib_audio_dsp)


set(PIPELINE_DIR ${REL_PATH_TO_BUILD}/dsp_pipeline_default)

set(APP_INCLUDES .. src ${PIPELINE_DIR})

file(GLOB C_SRC CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_LIST_DIR} src/*.c)


list(APPEND APP_C_SRCS
    ${C_SRC} ${PIPELINE_DIR}/adsp_generated_auto.c
)



# Read the JSON config file
file(READ "${CMAKE_CURRENT_LIST_DIR}/config.json" JSON_CONTENT)

# Function to extract JSON array into a CMake list
function(json_array_to_list json_content key out_var)
    string(JSON len LENGTH "${json_content}" ${key})
    math(EXPR max_index "${len} - 1")
    set(result)
    foreach(i RANGE 0 ${max_index})
        string(JSON val GET "${json_content}" ${key} ${i})
        list(APPEND result "${val}")
    endforeach()
    set(${out_var} "${result}" PARENT_SCOPE)
endfunction()

# Extract arrays from JSON
json_array_to_list("${JSON_CONTENT}" FS         FSes)
json_array_to_list("${JSON_CONTENT}" N_CHANS    N_CHANSes)
json_array_to_list("${JSON_CONTENT}" FRAME_SIZE FRAME_SIZES)
json_array_to_list("${JSON_CONTENT}" N_THREADS N_THREADes)

# Use the extracted values
foreach(FRAME_SIZE ${FRAME_SIZES})
    foreach(FS ${FSes})
        foreach(N_CHANS ${N_CHANSes})
            foreach(N_THREADS ${N_THREADes})
                set(APP_COMPILER_FLAGS_${FRAME_SIZE}_${FS}_${N_CHANS}_${N_THREADS}
                    -DFRAME_SIZE=${FRAME_SIZE}
                    -DFS=${FS}
                    -DN_CHANS=${N_CHANS}
                    -DN_THREADS=${N_THREADS}
                    ${COMMON_APP_COMPILER_FLAGS})
            endforeach()
        endforeach()
    endforeach()
endforeach()

set(XMOS_SANDBOX_DIR ${CMAKE_SOURCE_DIR}/../../../..)

XMOS_REGISTER_APP()
