

.PHONY: check update

check:
	pyright audio_dsp --skipunannotated --level warning
	black audio_dsp --check

update:
	black audio_dsp
