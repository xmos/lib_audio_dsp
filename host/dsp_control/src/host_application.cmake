# Building main application here

set( APP_NAME  dsp_host )

set(COMMON_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils/utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils/types_support.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils/platform_support.cpp
    ${CMAKE_CURRENT_LIST_DIR}/command/command.cpp
    ${CMAKE_CURRENT_LIST_DIR}/options/options.cpp
)
set(COMMON_INCLUDES
    ${CMAKE_CURRENT_LIST_DIR}/utils
    ${CMAKE_CURRENT_LIST_DIR}/device
    ${CMAKE_CURRENT_LIST_DIR}/command
    ${CMAKE_CURRENT_LIST_DIR}/options
    ${DEVICE_CONTROL_PATH}/api
)

add_executable( ${APP_NAME})

# Add options for different compilers
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options( ${APP_NAME}
        PRIVATE
            -WX
    )
else()
    target_compile_options( ${APP_NAME}
        PRIVATE
            -Werror
            -g
    )
endif()

target_sources( ${APP_NAME}
    PRIVATE
        ${COMMON_SOURCES}
)
target_include_directories( ${APP_NAME}
    PUBLIC
        ${COMMON_INCLUDES}
)

if (NOT ${CMAKE_SYSTEM_NAME} STREQUAL Windows)
target_link_libraries(dsp_host
    PUBLIC
        dl
)
target_link_options(dsp_host
    PRIVATE
        -rdynamic
)
endif() # not windows
