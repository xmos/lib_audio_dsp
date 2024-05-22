# Copyright 2021-2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import pytest
import subprocess
from pathlib import Path
import re

from audio_dsp.design import pipeline
from stages.dummy import Dummy


def gen_dummy_pipeline():
    p = pipeline.Pipeline(1, identifier="dummy")
    t = p.add_thread()
    d = t.stage(Dummy, p.i, label="dummy")
    p.set_outputs(d.o)
    pipeline.generate_dsp_main(p, out_dir=Path(__file__).parent / "build/dummy_pipeline")


def pytest_configure():
    gen_dummy_pipeline()
    if not Path("bin").exists():
        subprocess.run(["cmake", "-B", "build"], check=True)
    subprocess.run(["cmake", "--build", "build"], check=True)

## Begin pytest magic to convert xe files into tests

def pytest_collect_file(parent, path):
    """Custom collection function to inform pytest that xe files contain tests."""
    if path.ext == ".xe":
        return UnityTestSource.from_parent(parent, path=Path(path))

class UnityTestSource(pytest.File):
    """
    Each xe file contains 1 pytest test.
    """
    def collect(self):
        yield UnityTestExecutable.from_parent(self, xe=self.path, name=self.path.stem)


class UnityTestExecutable(pytest.Item):
    """
    Run the xe file in xsim, this is the work of the test.
    """
    def __init__(self, xe, **kwargs):
        super().__init__(**kwargs)
        self.xe=xe
        self.fail_reason=[]

    def runtest(self):
        """
        fancy test output processing.
        """
        proc = subprocess.run(["xsim", self.xe], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.add_report_section("call", "stdout", proc.stdout)
        unity_result_pattern=r"^(?P<path>[^\n:]+):(?P<line>\d+):(?P<name>[^:]+):(?P<status>PASS|FAIL)(: (?P<message>.*))?$"
        unlikely_repl = "unlikely_repl"

        result = [i for i in re.finditer(unity_result_pattern, proc.stdout, re.MULTILINE)]
        all_out = [i for i in re.sub(unity_result_pattern, unlikely_repl, proc.stdout, flags=re.MULTILINE).split(unlikely_repl)]

        for match, output in zip(result, all_out):
            file, line, test_name, status, message = match.group("path", "line", "name", "status", "message")
            fail_reason = f"{test_name}:{line}:{message}"
            self.add_report_section("call", f"{status} {test_name}", output + "\n" + fail_reason)
            if status == "FAIL":
                self.fail_reason.append(fail_reason)
        if proc.returncode:
            raise UnityTestException

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, UnityTestException):
            return "Failure summary:\n\t" + "\n\t".join(self.fail_reason)
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.path, 0, self.xe.stem


class UnityTestException(Exception):
    pass

## End pytest magic
