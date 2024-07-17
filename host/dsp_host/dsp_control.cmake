set(DEVICE_CONTROL_PATH ${CMAKE_SOURCE_DIR}/src/device_control)
message(STATUS, ${DEVICE_CONTROL_PATH})
include(${CMAKE_CURRENT_LIST_DIR}/src/host_drivers.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/src/host_application.cmake)
