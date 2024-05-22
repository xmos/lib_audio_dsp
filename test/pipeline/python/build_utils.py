# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""
Utility functions for building and running the application within
the jupyter notebook
"""
import subprocess
import ipywidgets as widgets
from IPython import display
from pathlib import Path
import shutil
import os

def build(source_dir, build_dir, target):
    """
    Attempt to build and xrun the application
    """
    print("Build and run - output will be in the terminal if it is not displayed below\r")
    cache = build_dir / "CMakeCache.txt"
    makefile = build_dir / "Makefile"
    ninjabuild = build_dir / "build.ninja"
    if (not cache.exists()) or not (makefile.exists() or ninjabuild.exists()):
        print("Configuring...\r")
        if cache.exists():
            # Generator is already known by cmake
            ret = subprocess.run([*(f"cmake -S {source_dir} -B {build_dir}".split())])
        else:
            # need to configure, default to Ninja because its better
            generator = "Ninja" if shutil.which("ninja") else "Unix Makefiles"
            ret = subprocess.run([*(f"cmake -S {source_dir} -B {build_dir} -G".split()), generator])
        if ret.returncode:
            print("Configuring failed, check log for details\r")
            assert(0)

    print("Compiling...\r")
    if os.name == "nt":
        ret = subprocess.run(f"cmake --build {build_dir} --target {target}".split())
    else:
        ret = subprocess.run(f"cmake --build {build_dir} --target {target}".split())
    if ret.returncode:
        print("ERROR: Building failed, check log for details\r")
        assert(0)
