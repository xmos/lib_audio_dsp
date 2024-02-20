#! /bin/bash
#
# FOR USE IN CI
#
# We need to install some extra python dependencies into the docker file
# for xmosdoc to run as we use autodoc to document our python APIs and
# have a script which runs and needs dependencies

set -ex

# create venv as we don't have permission to install packages to the system python
python -m venv .doc_venv
source .doc_venv/bin/activate
pip install -e /xmosdoc  # xmosdoc is cloned in this directory of the docker container
pip install -e /build/python  # Install lib_audio_dsp into the docker container so that
                              # autodoc can document its APIs
pip install docstring-inheritance # used for inheriting docstrings in classes
xmosdoc -dvvv clean html pdf linkcheck
