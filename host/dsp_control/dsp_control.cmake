## Rasbian version that we currently use 3.13  is the latest possible version of cmake
## This cmake file will not be integrated with the top-level cmake structure till this is resolved
cmake_minimum_required(VERSION 3.13)

option(TESTING "If set, cmake will build tests" OFF)
# Turn the option ON to use clang, you may need to change the path to your clang compiler
option(USE_CLANG "If set, cmake will use clang insted of gcc" OFF)
if(USE_CLANG)
    set(CMAKE_CXX_COMPILER "/usr/bin/clang++" CACHE STRING "C++ compiler" FORCE)
    set(CMAKE_C_COMPILER "/usr/bin/clang" CACHE STRING "C compiler" FORCE)
endif()

if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    include(GenerateExportHeader)
endif()

enable_language(CXX C)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Only Release mode is allowed" FORCE)

project(host_dsp_control)

set(DEVICE_CONTROL_PATH ${CMAKE_SOURCE_DIR}/src/device_control)

include(src/host_drivers.cmake)
include(src/host_application.cmake)
