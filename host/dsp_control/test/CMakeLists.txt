
add_library(command_map_dummy SHARED command_map_dummy.cpp)
target_link_options(command_map_dummy PRIVATE -fPIC)
target_include_directories(command_map_dummy PUBLIC ${DEVICE_CONTROL_PATH}/api)

add_library(device_dummy SHARED)
target_sources(device_dummy
    PRIVATE
        device_dummy.cpp
)
target_link_options(device_dummy PRIVATE -fPIC)
target_include_directories(device_dummy 
    PUBLIC 
        ${CMAKE_SOURCE_DIR}/src/device
        ${DEVICE_CONTROL_PATH}/api
)

if(WIN32)
    generate_export_header(command_map_dummy)
    generate_export_header(device_dummy)
endif()
