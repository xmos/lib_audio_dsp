set(LIB_NAME lib_audio_dsp)
set(LIB_VERSION 0.1.0)
set(LIB_INCLUDES api)
set(LIB_DEPENDENT_MODULES "lib_xcore_math(xcommon_cmake)" "lib_logging")
set(LIB_COMPILER_FLAGS -Os -Wall -Werror -g -mcmodel=large)
set(LIB_OPTIONAL_HEADERS adsp_generated.h)

XMOS_REGISTER_MODULE()
