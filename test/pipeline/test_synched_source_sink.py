# Copyright 2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from pathlib import Path
from filelock import FileLock
import subprocess
import pytest
import json

from audio_dsp.design.pipeline import Pipeline, generate_dsp_main
from python import build_utils, run_pipeline_xcoreai, audio_helpers
from stages.wait import Wait
from python.run_pipeline_xcoreai import FORCE_ADAPTER_ID

APP_NAME = "app_synched_source_sink"
APP_DIR = Path(__file__).parent / APP_NAME
VCD_DIR = APP_DIR / "vcd"
BUILD_DIR = APP_DIR / "build"

TEST_CONFIG = APP_DIR / "config.json"

TEST_PARAMS = json.loads(TEST_CONFIG.read_text())

@pytest.fixture
def xfail_selected(request):
    """
    Mark some known failures which dont meet timing.
    """
    get = lambda x: request.getfixturevalue(x)

    if (get("threads"), get("n_chans"), get("frame_size"), get("fs")) in [
        (1, 8, 16, 96000),
        (5, 8, 16, 96000),
        (1, 8, 1, 96000)   # would pass with wait_ratio=0.3
    ]:
        request.node.add_marker(pytest.mark.xfail(reason="Current benchmarking shows this should fail", strict=True))


@pytest.mark.parametrize("fs", TEST_PARAMS["FS"])
@pytest.mark.parametrize("frame_size", TEST_PARAMS["FRAME_SIZE"])
@pytest.mark.parametrize("n_chans", TEST_PARAMS["N_CHANS"])
@pytest.mark.parametrize("threads", TEST_PARAMS["N_THREADS"])
@pytest.mark.usefixtures('xfail_selected')
def test_synched_source_sync(fs, frame_size, n_chans, threads):
    """
    Basic benchmarking to find scenarios that fail.

    Test configuration is defined in config.json. This is also read by
    CMakeLists.txt to ensure the tests align with the DUT application.

    A special "Wait" stage has been written to consume a fixed ratio of each
    DSP thread time. The remaining time is available for the DSP thread to do
    control and communication.
    """
    p, i = Pipeline.begin(n_chans,fs=fs, frame_size=frame_size)
    wait_ratio = 0.7 # ratio of thread time spent doing DSP
    i = p.stage(Wait, i, wait_ratio=wait_ratio)
    for _ in range(threads - 1):
        p.next_thread()
        i = p.stage(Wait, i, wait_ratio=wait_ratio)
    p.set_outputs(i)
    config = f"{frame_size}_{fs}_{n_chans}_{threads}"

    BUILD_DIR.mkdir(exist_ok=True)
    VCD_DIR.mkdir(exist_ok=True)
    with FileLock(build_utils.BUILD_LOCK):
        generate_dsp_main(p, out_dir = BUILD_DIR / "dsp_pipeline_default")
        build_utils.build(APP_DIR, BUILD_DIR, f"app_synched_source_sink_{config}")
    vcd_file = VCD_DIR / f"{config}.vcd"
    xscope_file = VCD_DIR / f"{config}.xmt"
    app = APP_DIR / "bin" / config / f"{APP_NAME}_{config}.xe"
    subprocess.run(["xsim", "--xscope", f"-offline {xscope_file}", "--vcd-tracing", f"-o {vcd_file} -tile tile[0] -cores -instructions", app], check=True, timeout=60)




