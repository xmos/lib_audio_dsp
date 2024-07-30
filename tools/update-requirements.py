# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

"""
Wrapper around pip-tools to make life easy, Python for portability
each <file.in> in the current folder will produce a compiled <file.txt> on top-level folder
"""

import os
import sys
import subprocess
from pathlib import Path
import pip

cwd = Path(__file__).parent.absolute()
repo_root = cwd.parent.absolute()


def get_header_text():
    py_major, py_minor = sys.version_info.major, sys.version_info.minor
    pip_version_major = (pip.__version__).split(".", maxsplit=2)[0]
    header_text = f"# python_version {py_major}.{py_minor}\n"
    header_text += f"# pip_version {pip_version_major}.*\n"
    return header_text


def main():
    # configure pip-tools header comment
    os.environ["CUSTOM_COMPILE_COMMAND"] = f"./tools/update-requirements.py"
    
    # glob each .in and produce a .txt
    in_files = cwd.glob("*.in")
    for req_in in in_files:
        req_in = req_in.relative_to(repo_root) # needed for relative path output
        req_dst = (repo_root / req_in.with_suffix(".txt").name).relative_to(repo_root)
        opts = "--no-emit-trusted-host --no-emit-index-url"
        cmd = f"pip-compile {req_in} -o {req_dst} {opts}"
        subprocess.run(cmd, shell=True, check=True, cwd=repo_root, timeout=200)
        # add header to req_dst
        header_text = get_header_text()
        req_dst.write_text(header_text + req_dst.read_text())


if __name__ == "__main__":
    main()
