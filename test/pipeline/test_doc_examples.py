# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.



from pathlib import Path
import pytest
import sys
from subprocess import run
from filelock import FileLock
from python import build_utils

EXAMPLES_DIR = Path(__file__).parent/"doc_examples"


EXAMPLES = list(EXAMPLES_DIR.glob("*.py"))

@pytest.mark.parametrize("example", EXAMPLES, ids=[e.name for e in EXAMPLES])
def test_doc_examples(example):
    """Run all the Python scripts in doc_examples/"""
    with FileLock(build_utils.PIPELINE_BUILD_LOCK):
        run([sys.executable, example], check=True)
    
    
