# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from python import run_pipeline_xcoreai
import numpy as np
import scipy.signal as spsig
from pathlib import Path
import pytest


def pytest_configure(config):
    run_pipeline_xcoreai.FORCE_ADAPTER_ID = config.getoption("--adapter-id")

def pytest_addoption(parser):
    parser.addoption(
        "--adapter-id", action="store", default=None, help="Force tests to use specific adapter"
    )

def pytest_collection_modifyitems(items, config):
    for item in items:
        if not any(item.iter_markers("group0")):
            item.add_marker("group1")