

.PHONY: check update

check:
	pyright audio_dsp --skipunannotated --level warning
	ruff check
	ruff format --check

update:
	ruff check --fix
	ruff format

dev:
	make update
	make check
