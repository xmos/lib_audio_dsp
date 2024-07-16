# This file provides input to the Doxyfile configuration
# Note that xmosdoc defaults will take precedence over
# these values, edit doxygen overrides in settings.yml
# for higher priority settings

PROJECT_NAME = Lib audio dsp
PROJECT_BRIEF = "Audio DSP for the xcore"

INPUT += ../lib_audio_dsp/api/dsp \
         ../lib_audio_dsp/api/stages/adsp_control.h \
         ../lib_audio_dsp/api/stages/adsp_module.h \
         ../lib_audio_dsp/api/stages/adsp_pipeline.h \
         ./programming_guide/gen/control_gen/common

INCLUDE_PATH = ../lib_audio_dsp/api \
               ./programming_guide/gen/control_gen/common


PREDEFINED = DWORD_ALIGNED= DECLARE_JOB(a,b)=

