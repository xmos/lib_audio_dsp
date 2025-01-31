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
         ../lib_audio_dsp/api/control/helpers.h \
         ../lib_audio_dsp/api/control/reverb.h \
         ../lib_audio_dsp/api/control/reverb_plate.h \
         ../lib_audio_dsp/api/control/biquad.h \
         ../lib_audio_dsp/api/control/signal_chain.h \
         ../lib_audio_dsp/api/control/drc.h \
         ../lib_audio_dsp/api/control/signal_chain.h \

INCLUDE_PATH = ../lib_audio_dsp/api \


PREDEFINED = DWORD_ALIGNED= DECLARE_JOB(a,b)=

