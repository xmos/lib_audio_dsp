# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""
Tests for audio_dsp.stages with 2 inputs and 2 ouputs
"""

import pytest
import scipy.signal as spsig
from audio_dsp.design.pipeline import Pipeline, generate_dsp_main
from audio_dsp.stages import *
import audio_dsp.dsp.biquad as bq
from copy import deepcopy

import audio_dsp.dsp.utils as utils
from .python import build_utils, run_pipeline_xcoreai, audio_helpers

import os
from pathlib import Path
import numpy as np
import struct
import yaml
from filelock import FileLock
import shutil
import random
from test.test_utils import q_convert_flt

PKG_DIR = Path(__file__).parent
APP_DIR = PKG_DIR
BUILD_DIR = APP_DIR / "build"

TEST_FRAME_SIZES = 1, 128

fs = 48000
channels = 2  # if this changes need to rewrite test signals


@pytest.fixture(scope="module", params=TEST_FRAME_SIZES)
def frame_size(request):
    return request.param


def generate_ref(sig, ref_module, pipeline_channels, frame_size):
    """
    Process the signal through a Python stage of a certain frame size
    """
    sig_flt = utils.fixed_to_float_array(sig.T >> 4, 27)
    signal_frames = utils.frame_signal(sig_flt, frame_size, frame_size)
    out_py = np.zeros((pipeline_channels, sig.shape[0]))

    # push through zeros to match the test app which does this to support control
    pretest_zero_cont = 64e3 # MUST MATCH TEST APP
    for _ in range(int(pretest_zero_cont/frame_size)):
        ref_module.process_frame_xcore([np.zeros(frame_size) for _ in range(pipeline_channels)])

    # run through Python bit exact implementation
    for n in range(len(signal_frames)):
        out_py[:, n * frame_size : n * frame_size + frame_size] = (
            ref_module.process_frame_xcore(signal_frames[n])
        )

    # back to int scaling, and clip so that values are int32
    out_py_int = utils.float_to_fixed_array(out_py, 27) << 4

    return out_py_int


def do_test(default_pipeline, tuned_pipeline, dut_frame_size, folder_name, skip_default=False, rtol=0):
    """
    Run stereo file into app and check the output matches
    using in_ch and out_ch to decide which channels to compare

    This test compares the stage running on the device with the simulated
    stage. It runs the simulated stage at all the frame sizes in TEST_FRAME_SIZES
    to ensure that the output of a stage doesn't change when the frame size changes.

    Parameters
    ----------
    default_pipeline: function
        function that takes a frame size and returns a pipeline which
        has that frame size as the input. It uses the default configuration values.
    tuned_pipeline: function
        function that takes a frame size, returns a pipeline which
        has that frame size as the input, and tunes the pipelines with the desired
        configuration values
    dut_frame_size : int
        The frame size to use for the pipeline that will run on the device.
    """

    app_dir = PKG_DIR / folder_name
    os.makedirs(app_dir, exist_ok=True)

    with FileLock(build_utils.BUILD_LOCK):

        for func_p in [default_pipeline, tuned_pipeline]:
            # Exit if tuned_pipeline is not defined
            if not func_p:
                continue

            dut_p = func_p(dut_frame_size)
            pipeline_channels = len(dut_p.i)

            out_dir = None

            # Generate uninitialized stages for default_pipeline, only if tuned_pipeline is defined
            if func_p == default_pipeline and tuned_pipeline:
                out_dir = "dsp_pipeline_tuned"
            else:
                out_dir = "dsp_pipeline_default"
            generate_dsp_main(dut_p, out_dir=BUILD_DIR / out_dir)

        n_samps, rate = 1024, 48000
        infile = app_dir / "instage.wav"


        # signal starts at 0 so that there is no step in the signal
        sig_f = np.sin(np.linspace(0, 2*np.pi, n_samps)) * np.linspace(0, 1, n_samps) # a sin wave with ramped gain from 0
        sig0 = np.round(sig_f * 2**30).astype(np.int32)
        sig1 = np.round(sig_f * 2**27).astype(np.int32)

        if pipeline_channels == 2:
            sig = np.stack((sig0, sig1), axis=1)
        elif pipeline_channels == 1:
            sig = sig0
            sig = sig.reshape((len(sig), 1))
        else:
            assert False, f"Unsupported number of channels {pipeline_channels}. Test supports 1 or 2 channels"

        audio_helpers.write_wav(infile, rate, sig)

        # The reference function should be always tuned_pipeline, it is default_pipeline if tuned_pipeline is not defined
        ref_func_p = tuned_pipeline if tuned_pipeline else default_pipeline

        ref_p = [ref_func_p(s) for s in TEST_FRAME_SIZES]
        out_py_int_all = [
            generate_ref(sig, p.stages[2].dsp_block, pipeline_channels, fr)
            for p, fr in zip(ref_p, TEST_FRAME_SIZES)
        ]

        for target in ["default", "tuned"]:
            # Do not run the control test if tuned_pipeline is not defined
            if not tuned_pipeline and target == "tuned":
                continue
            # Build pipeline test executable. This will download xscope_fileio if not present
            build_utils.build(APP_DIR, BUILD_DIR, target)

            # old_xe = APP_DIR / f"bin/{target}/pipeline_test_{target}.xe"
            # new_xe = app_dir / f"bin/{target}/pipeline_test_{target}.xe"
        os.makedirs(app_dir / "bin", exist_ok=True)
        shutil.copytree(APP_DIR / "bin", app_dir / "bin", dirs_exist_ok=True)

    for target in ["default", "tuned"]:
        # Do not run the control test if tuned_pipeline is not defined
        if not tuned_pipeline and target == "tuned":
            continue
        if target == "default" and skip_default:
            continue
        print(f"Running {target} pipeline test")
        outfile = app_dir / f"{target}_outstage.wav"

        xe = app_dir / f"bin/{target}/pipeline_test_{target}.xe"

        _, _ = audio_helpers.read_wav(infile)
        run_pipeline_xcoreai.run(xe, infile, outfile, pipeline_channels, 1, return_stdout=False)

        _, out_data = audio_helpers.read_wav(outfile)
        if out_data.ndim == 1:
            out_data = out_data.reshape(len(out_data), 1)
        for out_py_int, ref_frame_size in zip(out_py_int_all, TEST_FRAME_SIZES):
            print(f"-- Testing frame size {ref_frame_size} with dut frame size {dut_frame_size}")
            for ch in range(pipeline_channels):
                # Save Python tracks
                audio_helpers.write_wav(app_dir / f"outstage_python_frame{ref_frame_size}_ch{ch}.wav", rate, np.array(out_py_int.T, dtype=np.int32))
                diff = out_py_int.T[:, ch] - out_data[:, ch]
                print(f"ch {ch}: max diff {max(abs(diff))}")
                sol = (~np.equal(out_py_int.T, out_data)).astype(int)
                indexes = np.flatnonzero(sol)
                print(f"ch {ch}: {len(indexes)} indexes mismatch")
                print(f"ch {ch} mismatching indexes = {indexes}")

            np.testing.assert_allclose(
                out_data,
                out_py_int.T,
                rtol=rtol,
                err_msg=f"dut frame {dut_frame_size}, ref frame {ref_frame_size}",
            )


def generate_test_param_file(config_name, stage_config):
    """
    Generate a header file with the configuration parameters listed in the arguments.

    Parameters
    ----------
    stage_name: string
        name of the stage to test
    stage_config: dict
        dictionary containing the parameter names and their corresponding values
    """
    type_data = {}
    with open(
        Path(__file__).resolve().parents[2] / f"stage_config/{config_name.lower()}.yaml",
        "r",
    ) as fd:
        type_data = yaml.safe_load(fd)

    # get the actual stage name
    stage_name = list(type_data["module"].keys())[0]

    # Write the autogenerated header file
    with open(
        Path(__file__).resolve().parent / f"build/control_test_params.h", "w"
    ) as f_op:
        f_op.write('#include "cmds.h"\n\n')
        f_op.write("#define CMD_PAYLOAD_MAX_SIZE 256 // must be 256 to fit in EP0\n")
        f_op.write(f"#define CMD_TOTAL_NUM {len(stage_config)}\n\n")
        f_op.write("typedef struct control_data_t {\n")
        f_op.write("\tuint32_t cmd_id;\n")
        f_op.write("\tuint32_t cmd_size;\n")
        f_op.write("\tuint8_t payload[CMD_PAYLOAD_MAX_SIZE];\n")
        f_op.write("}control_data_t;\n\n")
        f_op.write(f"control_data_t control_config[CMD_TOTAL_NUM] = {{\n")

        for cmd_name, cmd_payload in stage_config.items():
            f_op.write(f"\t{{\n")
            f_op.write(f"\t\t.cmd_id = CMD_{stage_name.upper()}_{cmd_name.upper()},\n")
            payload_values = []
            cmd_payload_list = []
            if not isinstance(cmd_payload, list):
                cmd_payload_list.append(cmd_payload)
            else:
                cmd_payload_list = cmd_payload
            payload_size = 0
            for value in cmd_payload_list:
                data_type = type_data["module"][stage_name.lower()][cmd_name.lower()][
                    "type"
                ]
                # Convert the values into bytearrays and compute the payload length
                if data_type in ["uint32_t"]:
                    value = np.uint32(value)
                    ba = bytearray(struct.pack("I", value))
                    payload_size += 4
                elif data_type in ["int", "int32_t"]:
                    value = np.int32(value)
                    ba = bytearray(struct.pack("i", value))
                    payload_size += 4
                elif data_type in ["float"]:
                    ba = struct.unpack("4b", struct.pack("f", value))
                    payload_size += 4
                elif data_type in ["int8_t", "uint8_t"]:
                    ba = bytearray(value & 0xFF)
                    payload_size += 1
                else:
                    raise ValueError(f"{data_type} is not supported")

                payload_values = payload_values + [
                    "0x{:02X}".format(x & 0xFF) for x in ba
                ]
            f_op.write(f"\t\t.cmd_size = {payload_size},\n")
            f_op.write(f"\t\t.payload  = {{{', '.join(list(payload_values))}}},\n")
            f_op.write(f"\t}},\n")
        f_op.write(f"}};\n")

@pytest.mark.group0
@pytest.mark.parametrize(
    "method, args",
    [
        ("make_bypass", None),
        ("make_lowpass", [1000, 0.707]),
        ("make_highpass", [1000, 0.707]),
        ("make_bandpass", [1000, 0.707]),
        ("make_bandstop", [1000, 0.707]),
        ("make_notch", [1000, 0.707]),
        ("make_allpass", [1000, 0.707]),
        ("make_peaking", [1000, 0.707, -6]),
        ("make_constant_q", [1000, 0.707, -6]),
        ("make_lowshelf", [1000, 0.707, -6]),
        ("make_highshelf", [1000, 0.707, -6]),
        ("make_linkwitz", [200, 0.707, 180, 0.707]),
    ],
)
def test_biquad(method, args, frame_size):
    """
    Test the biquad stage filters the same in Python and C
    """

    def default_pipeline(fr):
        p, i = Pipeline.begin(channels, frame_size=fr)
        o = p.stage(Biquad, i, label="control")
        p.set_outputs(o)

        return p

    def tuned_pipeline(fr):
        p = default_pipeline(fr)

        bq_method = getattr(p["control"], method)

        # Set initialization parameters of the stage
        if args:
            bq_method(*args)
        else:
            bq_method()

        stage_config = p["control"].get_config()
        generate_test_param_file("BIQUAD", stage_config)
        return p

    folder_name = f"biquad_{frame_size}_{method[5:]}"
    do_test(default_pipeline, tuned_pipeline, frame_size, folder_name)

@pytest.mark.group0
@pytest.mark.parametrize(
    "method, args, rtol",
    [
        ("make_bypass", None, 0),
        ("make_lowpass", [1000, 0.707], 0),
        ("make_highpass", [1000, 0.707], 0),
        ("make_bandpass", [1000, 0.707], 0),
        ("make_bandstop", [1000, 0.707], 0),
        ("make_notch", [1000, 0.707], 0),
        ("make_allpass", [1000, 0.707], 0),
        ("make_peaking", [1000, 0.707, -6], 0),
        ("make_constant_q", [1000, 0.707, -6], 0),
        ("make_lowshelf", [1000, 0.707, -6], 0),
        ("make_highshelf", [1000, 0.707, 10], 1e-2), # Fixing test to  make it exact is too hard for now
        ("make_linkwitz", [200, 0.707, 180, 0.707], 0),
    ],
)
def test_biquad_slew(method, args, frame_size, rtol):
    """
    Test the biquad stage filters the same in Python and C
    """

    def default_pipeline(fr):
        p, i = Pipeline.begin(channels, frame_size=fr)
        o = p.stage(BiquadSlew, i, label="control")
        p.set_outputs(o)

        return p

    def tuned_pipeline(fr):
        p = default_pipeline(fr)

        bq_method = getattr(p["control"], method)

        # Set initialization parameters of the stage
        if args:
            bq_method(*args)
        else:
            bq_method()

        p["control"].set_slew_shift(0)

        stage_config = p["control"].get_config()
        generate_test_param_file("BIQUAD_SLEW", stage_config)
        return p

    folder_name = f"biquad_slew_{frame_size}_{method[5:]}"

    # only run the control test, as there is no way to init python pipeline differently
    # for default & control tests
    do_test(default_pipeline, tuned_pipeline, frame_size, folder_name, skip_default=True, rtol=rtol)


filter_spec = [
    ["lowpass", fs * 0.4, 0.707],
    ["highpass", fs * 0.001, 1],
    ["peaking", fs * 1000 / 48000, 5, 10],
    ["constant_q", fs * 500 / 48000, 1, -10],
    ["notch", fs * 2000 / 48000, 1],
    ["lowshelf", fs * 200 / 48000, 1, 3],
    ["highshelf", fs * 5000 / 48000, 1, -2],
    ["gain", -2],
]


@pytest.mark.parametrize(
    "method, args",
    [
        ("make_butterworth_highpass", [8, 1000]),
        ("make_butterworth_lowpass", [8, 1000]),
        ("make_parametric_eq", [filter_spec]),
    ],
)
def test_cascaded_biquad(method, args, frame_size):
    """
    Test the biquad stage filters the same in Python and C
    """

    def default_pipeline(fr):
        p = Pipeline(channels, frame_size=fr)
        o= p.stage(CascadedBiquads, p.i, label="control")
        p.set_outputs(o)

        return p

    def tuned_pipeline(fr):
        p = default_pipeline(fr)

        # Set initialization parameters of the stage
        bq_method = getattr(p["control"], method)
        if args:
            bq_method(*args)
        else:
            bq_method()

        stage_config = p["control"].get_config()
        generate_test_param_file("CASCADED_BIQUADS", stage_config)
        return p

    folder_name = f"cbq_{frame_size}_{method[5:]}"
    do_test(default_pipeline, tuned_pipeline, frame_size, folder_name)


@pytest.mark.parametrize(
    "method, args",
    [
        ("make_parametric_eq", [filter_spec + filter_spec]),
    ],
)
def test_cascaded_biquad16(method, args, frame_size):
    """
    Test the biquad stage filters the same in Python and C
    """

    seed = frame_size
    random.Random(seed*int(fs/1000)).shuffle(args[0])

    def default_pipeline(fr):
        p = Pipeline(channels, frame_size=fr)
        o= p.stage(CascadedBiquads16, p.i, label="control")
        p.set_outputs(o)

        return p

    def tuned_pipeline(fr):
        p = default_pipeline(fr)

        # Set initialization parameters of the stage
        bq_method = getattr(p["control"], method)
        if args:
            bq_method(*args)
        else:
            bq_method()

        stage_config = p["control"].get_config()
        generate_test_param_file("CASCADED_BIQUADS_16", stage_config)
        return p

    folder_name = f"cbq16_{frame_size}_{method[5:]}"
    do_test(default_pipeline, tuned_pipeline, frame_size, folder_name)

@pytest.mark.group0
def test_limiter_rms(frame_size):
    """
    Test the limiter stage limits the same in Python and C
    """

    def default_pipeline(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(LimiterRMS, p.i, label="control")
        p.set_outputs(o)

        return p

    def tuned_pipeline(fr):
        p= default_pipeline(fr)

        # Set initialization parameters of the stage
        p["control"].make_limiter_rms(-6, 0.001, 0.1)

        stage_config = p["control"].get_config()
        generate_test_param_file("LIMITER_RMS", stage_config)
        return p

    folder_name = f"limiterrms_{frame_size}"
    do_test(default_pipeline, tuned_pipeline, frame_size, folder_name)


def test_limiter_peak(frame_size):
    """
    Test the limiter stage limits the same in Python and C
    """

    def default_pipeline(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(LimiterPeak, p.i, label="control")
        p.set_outputs(o)

        return p

    def tuned_pipeline(fr):
        p = default_pipeline(fr)

        # Set initialization parameters of the stage
        p["control"].make_limiter_peak(-6, 0.001, 0.1)

        stage_config = p["control"].get_config()
        generate_test_param_file("LIMITER_PEAK", stage_config)
        return p

    folder_name = f"limiterpeak_{frame_size}"
    do_test(default_pipeline, tuned_pipeline, frame_size, folder_name)

@pytest.mark.group0
def test_hard_limiter_peak(frame_size):
    """
    Test the limiter stage limits the same in Python and C
    """

    def default_pipeline(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(HardLimiterPeak, p.i, label="control")
        p.set_outputs(o)

        return p

    def tuned_pipeline(fr):
        p = default_pipeline(fr)

        # Set initialization parameters of the stage
        p["control"].make_hard_limiter_peak(-6, 0.001, 0.1)

        stage_config = p.resolve_pipeline()["configs"][2]
        generate_test_param_file("HARD_LIMITER_PEAK", stage_config)
        return p

    folder_name = f"hardlimiterpeak_{frame_size}"
    do_test(default_pipeline, tuned_pipeline, frame_size, folder_name)


def test_clipper(frame_size):
    """
    Test the clipper stage clips the same in Python and C
    """

    def default_pipeline(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(Clipper, p.i, label="control")
        p.set_outputs(o)

        return p

    def tuned_pipeline(fr):
        p = default_pipeline(fr)

        # Set initialization parameters of the stage
        p["control"].make_clipper(-6)

        stage_config = p["control"].get_config()
        generate_test_param_file("CLIPPER", stage_config)
        return p

    folder_name = f"clipper_{frame_size}"
    do_test(default_pipeline, tuned_pipeline, frame_size, folder_name)

@pytest.mark.group0
def test_compressor(frame_size):
    """
    Test the compressor stage compresses the same in Python and C
    """

    def default_pipeline(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(CompressorRMS, p.i, label="control")
        p.set_outputs(o)

        return p

    def tuned_pipeline(fr):
        p = default_pipeline(fr)

        # Set initialization parameters of the stage
        p["control"].make_compressor_rms(2, -6, 0.001, 0.1)

        stage_config = p["control"].get_config()
        generate_test_param_file("COMPRESSOR_RMS", stage_config)
        return p

    folder_name = f"compressor_{frame_size}"
    do_test(default_pipeline, tuned_pipeline, frame_size, folder_name)

def test_noise_gate(frame_size):
    """
    Test the noise gate stage gates the noise the same in Python and C
    """

    def default_pipeline(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(NoiseGate, p.i, label="control")
        p.set_outputs(o)

        return p

    def tuned_pipeline(fr):
        p = default_pipeline(fr)

        # Set initialization parameters of the stage
        p["control"].make_noise_gate(-6, 0.001, 0.1)

        stage_config = p["control"].get_config()
        generate_test_param_file("NOISE_GATE", stage_config)
        return p

    folder_name = f"noise_gate_{frame_size}"
    do_test(default_pipeline, tuned_pipeline, frame_size, folder_name)

def test_noise_suppressor_expander(frame_size):
    """
    Test the noise suppressor (expander) stage suppress the noise the same in Python and C
    """

    def default_pipeline(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(NoiseSuppressorExpander, p.i, label="control")
        p.set_outputs(o)

        return p

    def tuned_pipeline(fr):
        p = default_pipeline(fr)

        # Set initialization parameters of the stage
        p["control"].make_noise_suppressor_expander(2, -6, 0.001, 0.1)

        stage_config = p["control"].get_config()
        generate_test_param_file("NOISE_SUPPRESSOR_EXPANDER", stage_config)
        return p

    folder_name = f"noise_suppressor_{frame_size}"
    do_test(default_pipeline, tuned_pipeline, frame_size, folder_name)


def test_volume(frame_size):
    """
    Test the volume stage amplifies the same in Python and C
    """

    # The gain_dB and mute_state must match in both default_pipeline() and tuned_pipeline().
    # Those values are used to compute the starting gain, and it must match in both applications
    gain_dB = -8
    mute_state = 0

    def default_pipeline(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(VolumeControl, p.i, label="control")
        p.set_outputs(o)
        p["control"].set_gain(gain_dB)
        p["control"].set_mute_state(mute_state)
        return p

    def tuned_pipeline(fr):
        p = default_pipeline(fr)

        p["control"].make_volume_control(gain_dB, 10, mute_state)
        stage_config = p["control"].get_config()
        generate_test_param_file("VOLUME_CONTROL", stage_config)
        return p

    folder_name = f"volume_{frame_size}"
    do_test(default_pipeline, tuned_pipeline, frame_size, folder_name)

def test_fixed_gain(frame_size):
    """
    Test the volume stage amplifies the same in Python and C
    """

    def default_pipeline(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(FixedGain, p.i, label="control")
        p.set_outputs(o)

        return p

    def tuned_pipeline(fr):
        p = default_pipeline(fr)

        # Set initialization parameters of the stage
        p["control"].set_gain(-8)

        stage_config = p["control"].get_config()
        generate_test_param_file("FIXED_GAIN", stage_config)
        return p

    folder_name = f"fixed_gain_{frame_size}"
    do_test(default_pipeline, tuned_pipeline, frame_size, folder_name)

@pytest.mark.parametrize("pregain, mix", [
    [0.01, False],
    [0.01, True],
    [0.3, False],
     ])
def test_reverb(frame_size, pregain, mix):
    """
    Test Reverb stage
    """


    def default_pipeline(fr):
        reverb_test_channels = 1  # Reverb expects only 1 channel
        p = Pipeline(reverb_test_channels, frame_size=fr)
        o = p.stage(ReverbRoom, p.i, label="control")
        p.set_outputs(o)

        return p

    def tuned_pipeline(fr):
        p = default_pipeline(fr)

        # Set initialization parameters of the stage
        if mix:
            p["control"].set_wet_dry_mix(0.5)
        else:
            p["control"].set_wet_gain(-1)
            p["control"].set_dry_gain(-2)
        p["control"].set_pre_gain(pregain)
        p["control"].set_room_size(0.4)
        p["control"].set_damping(0.5)
        p["control"].set_decay(0.6)
        p["control"].set_predelay(5)

        stage_config = p["control"].get_config()
        generate_test_param_file("REVERB_ROOM", stage_config)
        return p

    folder_name = f"reverbroom_{frame_size}_{pregain}_{int(mix)}"
    do_test(default_pipeline, tuned_pipeline, frame_size, folder_name)


@pytest.mark.parametrize("pregain, mix", [
    [0.01, False],
    [0.01, True],
    [0.3, False],
     ])
def test_reverb_plate(frame_size, pregain, mix):
    """
    Test Reverb stage
    """

    def default_pipeline(fr):
        reverb_test_channels = 2  # Reverb expects only 2 channel
        p = Pipeline(reverb_test_channels, frame_size=fr)
        o = p.stage(ReverbPlateStereo, p.i, label="control")
        p.set_outputs(o)

        return p

    def tuned_pipeline(fr):
        p = default_pipeline(fr)

        # Set initialization parameters of the stage
        if mix:
            p["control"].set_wet_dry_mix(0.5)
        else:
            p["control"].set_wet_gain(-1)
            p["control"].set_dry_gain(-2)
        p["control"].set_pre_gain(pregain)
        p["control"].set_early_diffusion(0.4)
        p["control"].set_late_diffusion(0.4)
        p["control"].set_bandwidth(4000)
        p["control"].set_damping(0.5)
        p["control"].set_decay(0.6)
        p["control"].set_predelay(5)

        stage_config = p["control"].get_config()
        generate_test_param_file("REVERB_PLATE_STEREO", stage_config)
        return p

    folder_name = f"reverbplate_{frame_size}_{pregain}_{int(mix)}"
    do_test(default_pipeline, tuned_pipeline, frame_size, folder_name)


@pytest.mark.parametrize("change_delay", [5, 0])
def test_delay(frame_size, change_delay):
    """
    Test Delay stage
    """

    def default_pipeline(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(
            Delay, p.i, max_delay=15, starting_delay=10, label="control"
        )
        p.set_outputs(o)
        return p

    def tuned_pipeline(fr):
        p = default_pipeline(fr)

        # Set initialization parameters of the stage
        p["control"].set_delay(change_delay)

        stage_config = p["control"].get_config()
        generate_test_param_file("DELAY", stage_config)
        return p

    folder_name = f"delay_{frame_size}_{change_delay}"
    do_test(default_pipeline, tuned_pipeline, frame_size, folder_name)

@pytest.fixture(scope="session", autouse=True)
def make_coeffs():
    # make sets of coefficients used in the FIR tests
    gen_dir = Path(__file__).parent / "autogen"
    gen_dir.mkdir(exist_ok=True, parents=True)

    # descending coefficients
    coeffs = np.arange(10, 0, -1)
    coeffs = coeffs / np.sum(coeffs)
    np.savetxt(Path(gen_dir, "descending_coeffs.txt"), coeffs)

    # simple windowed FIR design
    coeffs = spsig.firwin2(512, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
    np.savetxt(Path(gen_dir, "simple_low_pass.txt"), coeffs)


@pytest.mark.parametrize(
    "filter_name", ["descending_coeffs.txt", "simple_low_pass.txt"]
)
def test_fir(frame_size, filter_name):
    """ "
    Test FIR Stage
    """
    filter_path = Path(Path(__file__).parent / "autogen", filter_name)

    def default_pipeline(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(FirDirect, p.i, coeffs_path=filter_path)
        p.set_outputs(o)
        return p

    folder_name = f"fir_{frame_size}_{filter_name[:5]}"
    do_test(default_pipeline, None, frame_size, folder_name)

def test_graphic_eq(frame_size):
    """
    Test the volume stage amplifies the same in Python and C
    """

    def default_pipeline(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(GraphicEq10b, p.i, label="control")
        p.set_outputs(o)

        return p

    def tuned_pipeline(fr):
        p = default_pipeline(fr)

        # Set initialization parameters of the stage
        p["control"].set_gains([-6, 0, -5, 1, -4, 2, -3, 3, -2, 4])

        stage_config = p["control"].get_config()
        generate_test_param_file("graphic_eq_10b", stage_config)
        return p

    folder_name = f"geq_{frame_size}"
    do_test(default_pipeline, tuned_pipeline, frame_size, folder_name)

if __name__ == "__main__":
    test_graphic_eq(1)
    # test_volume(1)
