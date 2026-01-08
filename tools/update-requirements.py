# Copyright 2024-2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

"""
Wrapper around pip-tools to make life easy, Python for portability
each <file.in> in the current folder will produce a compiled <file.txt> on top-level folder

Usage:

    python tools/update-requirements.py [package1 package2 ...]

Named packages are optional and will be upgraded to the latest version.
"""

import os
import sys
import subprocess
from pathlib import Path
import pip
from contextlib import contextmanager, ExitStack
from tempfile import TemporaryDirectory

cwd = Path(__file__).parent.absolute()
repo_root = cwd.parent.absolute()


def get_header_text():
    py_major, py_minor = sys.version_info.major, sys.version_info.minor
    pip_version_major = (pip.__version__).split(".", maxsplit=2)[0]
    header_text = f"# python_version {py_major}.{py_minor}\n"
    header_text += f"# pip_version {pip_version_major}.*\n"
    return header_text

@contextmanager
def gen_valid_constraints(req_file):
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        constraint_file = temp_dir / "constraints.txt"

        with open(req_file, "r") as f:
            with open(constraint_file, "w") as c:
                for line in f:
                    if not line.startswith("-e") and not "git" in line:
                        c.write(line)
        yield constraint_file


def gen(req_in, constraints=tuple(), upgrade=tuple()):
    with ExitStack() as stack:
        upgrade = " ".join(f"--upgrade-package {u}" for u in upgrade)
        constraints = [stack.enter_context(gen_valid_constraints(c)) for c in constraints]
        req_in = req_in.relative_to(repo_root) # needed for relative path output
        req_dst = (repo_root / req_in.with_suffix(".txt").name).relative_to(repo_root)
        opts = "--no-emit-trusted-host --no-emit-index-url"
        constraints = " ".join(f"-c {c}" for c in constraints)
        cmd = f"pip-compile {req_in} {constraints} {upgrade} -o {req_dst} {opts}"
        subprocess.run(cmd, shell=True, check=True, cwd=repo_root, timeout=200)
        # add header to req_dst
        header_text = get_header_text()
        req_dst.write_text(header_text + req_dst.read_text())
        return req_dst


def main():
    upgrade = sys.argv[1:]

    # configure pip-tools header comment
    os.environ["CUSTOM_COMPILE_COMMAND"] = f"./tools/update-requirements.py"

    req_in = cwd/"requirements.in"
    reqs = gen(req_in, upgrade=upgrade)
    gen(cwd/"requirements-format.in", constraints=(reqs,))

if __name__ == "__main__":
    main()
