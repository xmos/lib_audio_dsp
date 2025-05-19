
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

@pytest.mark.parametrize("fs", TEST_PARAMS["FS"])
@pytest.mark.parametrize("frame_size", TEST_PARAMS["FRAME_SIZE"])
@pytest.mark.parametrize("n_chans", TEST_PARAMS["N_CHANS"])
@pytest.mark.parametrize("threads", TEST_PARAMS["N_THREADS"])
def test_synched_source_sync(fs, frame_size, n_chans, threads):
    p, i = Pipeline.begin(n_chans,fs=fs, frame_size=frame_size)
    i = p.stage(Wait, i)
    for _ in range(threads - 1):
        p.next_thread()
        i = p.stage(Wait, i)
    p.set_outputs(i)
    config = f"{frame_size}_{fs}_{n_chans}_{threads}"

    BUILD_DIR.mkdir(exist_ok=True)
    VCD_DIR.mkdir(exist_ok=True)
    with FileLock(build_utils.SYNCHED_SOURCE_SINK_BUILD_LOCK):
        generate_dsp_main(p, out_dir = BUILD_DIR / "dsp_pipeline_default")
        build_utils.build(APP_DIR, BUILD_DIR, f"app_synched_source_sink_{config}")
    vcd_file = VCD_DIR / f"{config}.vcd"
    app = APP_DIR / "bin" / config / f"{APP_NAME}_{config}.xe"
    subprocess.run(["xsim", app, "--vcd-tracing", f"-o {vcd_file} -tile tile[0] -cores -instructions"], check=True)



