# This file provides input to the Doxyfile configuration
# Note that xmosdoc defaults will take precedence over
# these values, edit doxygen overrides in settings.yml
# for higher priority settings

PROJECT_NAME = Lib audio dsp
PROJECT_BRIEF = "Audio DSP for the xcore"

INPUT += ../lib_audio_dsp/api

PREDEFINED = DWORD_ALIGNED= DECLARE_JOB(a,b)=

