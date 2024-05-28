# Building host device_control drivers here

# Build device_control_host for USB

add_library(device_control_host_usb INTERFACE)

# Discern OS for libusb library location
if ((${CMAKE_SYSTEM_NAME} MATCHES "Darwin") AND (${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64"))
    target_link_directories(device_control_host_usb INTERFACE "${DEVICE_CONTROL_PATH}/host/libusb/OSX64")
    set(libusb-1.0_INCLUDE_DIRS "${DEVICE_CONTROL_PATH}/host/libusb/OSX64")
    set(LINK_LIBS usb-1.0.0)
elseif ((${CMAKE_SYSTEM_NAME} MATCHES "Darwin") AND (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64"))
    target_link_directories(device_control_host_usb INTERFACE "${DEVICE_CONTROL_PATH}/host/libusb/OSXARM")
    set(libusb-1.0_INCLUDE_DIRS "${DEVICE_CONTROL_PATH}/host/libusb/OSXARM")
    set(LINK_LIBS usb-1.0.0)
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    find_package(PkgConfig)
    pkg_check_modules(libusb-1.0 REQUIRED libusb-1.0)
    set(LINK_LIBS usb-1.0)
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    add_compile_definitions(nologo WAll WX- O2 EHa)
    target_link_directories(device_control_host_usb INTERFACE "${DEVICE_CONTROL_PATH}/host/libusb/Win32")
    set(libusb-1.0_INCLUDE_DIRS "${DEVICE_CONTROL_PATH}/host/libusb/Win32")
    set(LINK_LIBS libusb)
endif()

target_sources(device_control_host_usb
    INTERFACE
        ${DEVICE_CONTROL_PATH}/host/util.c
        ${DEVICE_CONTROL_PATH}/host/device_access_usb.c
)
target_include_directories(device_control_host_usb
    INTERFACE
        ${DEVICE_CONTROL_PATH}/api
        ${DEVICE_CONTROL_PATH}/host
        ${libusb-1.0_INCLUDE_DIRS}
)
target_compile_definitions(device_control_host_usb INTERFACE USE_USB=1)

target_link_libraries(device_control_host_usb
    INTERFACE
        ${LINK_LIBS}
)
add_library(device_control_host_usb ALIAS device_control_host_usb)

# Build a wrapper driver for USB

add_library(device_usb SHARED)
target_sources(device_usb
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/device/device_usb.cpp
)
target_include_directories(device_usb
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/device
        ${DEVICE_CONTROL_PATH}/host
)
target_link_libraries(device_usb
    PUBLIC
        device_control_host_usb
)

target_link_libraries(device_usb PRIVATE -fPIC)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_custom_command(
        TARGET device_usb
        POST_BUILD
        COMMAND ${CMAKE_INSTALL_NAME_TOOL} -change "/usr/local/lib/libusb-1.0.0.dylib" "@executable_path/libusb-1.0.0.dylib" ${CMAKE_BINARY_DIR}/"libdevice_usb.dylib"
    )
    add_custom_command(
        TARGET device_usb
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${DEVICE_CONTROL_PATH}/host/libusb/OSX64/libusb-1.0.0.dylib ${CMAKE_BINARY_DIR}
    )
endif()

# Build device_control_host for XSCOPE
# Exclude arm64 macOS
if (NOT (${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64"))

add_library(device_control_host_xscope INTERFACE)
target_sources(device_control_host_xscope
    INTERFACE
        ${DEVICE_CONTROL_PATH}/host/util.c
        ${DEVICE_CONTROL_PATH}/host/device_access_xscope.c
)
target_include_directories(device_control_host_xscope
    INTERFACE
        ${DEVICE_CONTROL_PATH}/api
        ${DEVICE_CONTROL_PATH}/host
        $ENV{XMOS_TOOL_PATH}/include
)

find_library(XSCOPE_ENDPOINT_LIB NAMES xscope_endpoint.so xscope_endpoint.lib
                                 PATHS $ENV{XMOS_TOOL_PATH}/lib)

target_link_libraries(device_control_host_xscope INTERFACE ${XSCOPE_ENDPOINT_LIB})

target_compile_definitions(device_control_host_xscope INTERFACE USE_XSCOPE=1)
add_library(device_control_host_xscope ALIAS device_control_host_xscope)

# Build a wrapper driver for xscope

add_library(device_xscope SHARED)
target_sources(device_xscope
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/device/device_xscope.cpp
)
target_include_directories(device_xscope
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/device
        ${DEVICE_CONTROL_PATH}/host
)
target_link_libraries(device_xscope
    PUBLIC
        device_control_host_xscope
)
target_link_libraries(device_xscope PRIVATE -fPIC)
endif()

