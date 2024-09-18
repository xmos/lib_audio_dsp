# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""
Tests for audio_dsp.stages with 2 inputs and 2 ouputs
"""

import pytest
import scipy.signal as spsig
from audio_dsp.design.pipeline import Pipeline, generate_dsp_main
from audio_dsp.stages import *

import audio_dsp.dsp.utils as utils
from python import build_utils, run_pipeline_xcoreai, audio_helpers

from pathlib import Path
import numpy as np
import struct
import yaml

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
    sig_flt = np.float64(sig.T) * 2**-31
    signal_frames = utils.frame_signal(sig_flt, frame_size, frame_size)
    out_py = np.zeros((pipeline_channels, sig.shape[0]))

    # run through Python bit exact implementation
    for n in range(len(signal_frames)):
        out_py[:, n * frame_size : n * frame_size + frame_size] = (
            ref_module.process_frame_xcore(signal_frames[n])
        )

    # back to int scaling, and clip so that values are int32
    out_py_int = out_py * 2**31

    return out_py_int


def do_test(make_p, tune_p, dut_frame_size):
    """
    Run stereo file into app and check the output matches
    using in_ch and out_ch to decide which channels to compare

    This test compares the stage running on the device with the simulated
    stage. It runs the simulated stage at all the frame sizes in TEST_FRAME_SIZES
    to ensure that the output of a stage doesn't change when the frame size changes.

    Parameters
    ----------
    make_p: function
        function that takes a frame size and returns a pipeline which
        has that frame size as the input. It uses the default configuration values.
    tune_p: function
        function that takes a frame size, returns a pipeline which
        has that frame size as the input, and tunes the pipelines with the desired
        configuration values
    dut_frame_size : int
        The frame size to use for the pipeline that will run on the device.
    """

    for func_p in [make_p, tune_p]:
        # Exit if tune_p is not defined
        if not func_p:
            continue

        dut_p = func_p(dut_frame_size)
        pipeline_channels = len(dut_p.i)

        out_dir = None

        # Generate uninitialized stages for make_p, only if tune_p is defined
        if func_p == make_p and tune_p:
            out_dir = "dsp_pipeline_uninitialized"
        else:
            out_dir = "dsp_pipeline_initialized"
        generate_dsp_main(dut_p, out_dir=BUILD_DIR / out_dir)

    n_samps, rate = 1024, 48000
    infile = "instage.wav"
    outfile = "outstage.wav"

    # The reference function should be always tune_p, it is make_p if tune_p is not defined
    ref_func_p = tune_p if tune_p else make_p

    ref_p = [ref_func_p(s) for s in TEST_FRAME_SIZES]
    sig0 = (
        np.linspace(-(2**26), 2**26, n_samps, dtype=np.int32) << 4
    )  # numbers which should be unmodified through pipeline
    # data formats
    sig1 = np.linspace(-(2**23), 2**23, n_samps, dtype=np.int32) << 4

    if pipeline_channels == 2:
        sig = np.stack((sig0, sig1), axis=1)
    elif pipeline_channels == 1:
        sig = sig0
        sig = sig.reshape((len(sig), 1))
    else:
        assert False, f"Unsupported number of channels {pipeline_channels}. Test supports 1 or 2 channels"

    audio_helpers.write_wav(infile, rate, sig)

    out_py_int_all = [
        generate_ref(sig, p.stages[2].dsp_block, pipeline_channels, fr)
        for p, fr in zip(ref_p, TEST_FRAME_SIZES)
    ]

    for target in ["default", "control_commands"]:
        # Do not run the control test if tune_p is not defined
        if not tune_p and target == "control_commands":
            continue

        # Build pipeline test executable. This will download xscope_fileio if not present
        build_utils.build(APP_DIR, BUILD_DIR, target)

        xe = APP_DIR / f"bin/{target}/pipeline_test_{target}.xe"
        run_pipeline_xcoreai.run(xe, infile, outfile, pipeline_channels, 1)

        _, out_data = audio_helpers.read_wav(outfile)
        if out_data.ndim == 1:
            out_data = out_data.reshape(len(out_data), 1)
        for out_py_int, ref_frame_size in zip(out_py_int_all, TEST_FRAME_SIZES):
            for ch in range(pipeline_channels):
                # Save Python tracks
                audio_helpers.write_wav(outfile.replace(".wav", f"_python_ch{ch}.wav"), rate, np.array(out_py_int.T, dtype=np.int32))
                diff = out_py_int.T[:, ch] - out_data[:, ch]
                print(f"ch {ch}: max diff {max(abs(diff))}")
                sol = (~np.equal(out_py_int.T, out_data)).astype(int)
                indexes = np.flatnonzero(sol)
                print(f"ch {ch}: {len(indexes)} indexes mismatch")
                print(f"ch {ch} mismatching indexes = {indexes}")

            np.testing.assert_equal(
                out_py_int.T,
                out_data,
                err_msg=f"dut frame {dut_frame_size}, ref frame {ref_frame_size}",
            )


def generate_test_param_file(stage_name, stage_config):
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
        Path(__file__).resolve().parents[2] / f"stage_config/{stage_name.lower()}.yaml",
        "r",
    ) as fd:
        type_data = yaml.safe_load(fd)

    # Write the autogenerated header file
    with open(
        Path(__file__).resolve().parent / f"build/control_test_params.h", "w"
    ) as f_op:
        f_op.write('#include "cmds.h"\n\n')
        f_op.write("#define CMD_PAYLOAD_MAX_SIZE 256\n")
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

    def make_p(fr):
        p, i = Pipeline.begin(channels, frame_size=fr)
        o = p.stage(Biquad, i, label="control")
        p.set_outputs(o)

        return p

    def tune_p(fr):
        p = make_p(fr)

        bq_method = getattr(p["control"], method)

        # Set initialization parameters of the stage
        if args:
            bq_method(*args)
        else:
            bq_method()

        stage_config = p["control"].get_config()
        generate_test_param_file("BIQUAD", stage_config)
        return p

    do_test(make_p, tune_p, frame_size)


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

    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        o= p.stage(CascadedBiquads, p.i, label="control")
        p.set_outputs(o)

        return p

    def tune_p(fr):
        p = make_p(fr)

        # Set initialization parameters of the stage
        bq_method = getattr(p["control"], method)
        if args:
            bq_method(*args)
        else:
            bq_method()

        stage_config = p["control"].get_config()
        generate_test_param_file("CASCADED_BIQUADS", stage_config)
        return p

    do_test(make_p, tune_p, frame_size)


def test_limiter_rms(frame_size):
    """
    Test the limiter stage limits the same in Python and C
    """

    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(LimiterRMS, p.i, label="control")
        p.set_outputs(o)

        return p

    def tune_p(fr):
        p= make_p(fr)

        # Set initialization parameters of the stage
        p["control"].make_limiter_rms(-6, 0.001, 0.1)

        stage_config = p["control"].get_config()
        generate_test_param_file("LIMITER_RMS", stage_config)
        return p

    do_test(make_p, tune_p, frame_size)


def test_limiter_peak(frame_size):
    """
    Test the limiter stage limits the same in Python and C
    """

    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(LimiterPeak, p.i, label="control")
        p.set_outputs(o)

        return p

    def tune_p(fr):
        p = make_p(fr)

        # Set initialization parameters of the stage
        p["control"].make_limiter_peak(-6, 0.001, 0.1)

        stage_config = p["control"].get_config()
        generate_test_param_file("LIMITER_PEAK", stage_config)
        return p

    do_test(make_p, tune_p, frame_size)


def test_hard_limiter_peak(frame_size):
    """
    Test the limiter stage limits the same in Python and C
    """

    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(HardLimiterPeak, p.i, label="control")
        p.set_outputs(o)

        return p

    def tune_p(fr):
        p = make_p(fr)

        # Set initialization parameters of the stage
        p["control"].make_hard_limiter_peak(-6, 0.001, 0.1)

        stage_config = p.resolve_pipeline()["configs"][2]
        generate_test_param_file("HARD_LIMITER_PEAK", stage_config)
        return p

    do_test(make_p, tune_p, frame_size)


def test_clipper(frame_size):
    """
    Test the clipper stage clips the same in Python and C
    """

    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(Clipper, p.i, label="control")
        p.set_outputs(o)

        return p

    def tune_p(fr):
        p = make_p(fr)

        # Set initialization parameters of the stage
        p["control"].make_clipper(-6)

        stage_config = p["control"].get_config()
        generate_test_param_file("CLIPPER", stage_config)
        return p

    do_test(make_p, tune_p, frame_size)


def test_compressor(frame_size):
    """
    Test the compressor stage compresses the same in Python and C
    """

    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(CompressorRMS, p.i, label="control")
        p.set_outputs(o)

        return p

    def tune_p(fr):
        p = make_p(fr)

        # Set initialization parameters of the stage
        p["control"].make_compressor_rms(2, -6, 0.001, 0.1)

        stage_config = p["control"].get_config()
        generate_test_param_file("COMPRESSOR_RMS", stage_config)
        return p

    do_test(make_p, tune_p, frame_size)


def test_noise_gate(frame_size):
    """
    Test the noise gate stage gates the noise the same in Python and C
    """

    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(NoiseGate, p.i, label="control")
        p.set_outputs(o)

        return p

    def tune_p(fr):
        p = make_p(fr)

        # Set initialization parameters of the stage
        p["control"].make_noise_gate(-6, 0.001, 0.1)

        stage_config = p["control"].get_config()
        generate_test_param_file("NOISE_GATE", stage_config)
        return p

    do_test(make_p, tune_p, frame_size)


def test_noise_suppressor_expander(frame_size):
    """
    Test the noise suppressor (expander) stage suppress the noise the same in Python and C
    """

    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(NoiseSuppressorExpander, p.i, label="control")
        p.set_outputs(o)

        return p

    def tune_p(fr):
        p = make_p(fr)

        # Set initialization parameters of the stage
        p["control"].make_noise_suppressor_expander(2, -6, 0.001, 0.1)

        stage_config = p["control"].get_config()
        generate_test_param_file("NOISE_SUPPRESSOR_EXPANDER", stage_config)
        return p

    do_test(make_p, tune_p, frame_size)


def test_volume(frame_size):
    """
    Test the volume stage amplifies the same in Python and C
    """

    # The gain_dB and mute_state must match in both make_p() and tune_p().
    # Those values are used to compute the starting gain, and it must match in both applications
    gain_dB = -8
    mute_state = 0

    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(VolumeControl, p.i, label="control")
        p.set_outputs(o)
        p["control"].set_gain(gain_dB)
        p["control"].set_mute_state(mute_state)
        return p

    def tune_p(fr):
        p = make_p(fr)

        p["control"].make_volume_control(gain_dB, 10, mute_state)
        stage_config = p["control"].get_config()
        generate_test_param_file("VOLUME_CONTROL", stage_config)
        return p

    do_test(make_p, tune_p, frame_size)


def test_fixed_gain(frame_size):
    """
    Test the volume stage amplifies the same in Python and C
    """

    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(FixedGain, p.i, label="control")
        p.set_outputs(o)

        return p

    def tune_p(fr):
        p = make_p(fr)

        # Set initialization parameters of the stage
        p["control"].set_gain(-8)

        stage_config = p["control"].get_config()
        generate_test_param_file("FIXED_GAIN", stage_config)
        return p

    do_test(make_p, tune_p, frame_size)

@pytest.mark.parametrize("pregain, mix", [
    [0.01, False],
    [0.01, True],
    pytest.param([0.3, False], marks=pytest.mark.xfail(reason="Reverb can overflow for large values of pregain, tracked at LCD-297"))
     ])
def test_reverb(frame_size, pregain, mix):
    """
    Test Reverb stage
    """

    def make_p(fr):
        reverb_test_channels = 1  # Reverb expects only 1 channel
        p = Pipeline(reverb_test_channels, frame_size=fr)
        o = p.stage(ReverbRoom, p.i, label="control")
        p.set_outputs(o)

        return p

    def tune_p(fr):
        p = make_p(fr)

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

    do_test(make_p, tune_p, frame_size)

@pytest.mark.parametrize("change_delay", [5, 0])
def test_delay(frame_size, change_delay):
    """
    Test Delay stage
    """

    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(
            Delay, p.i, max_delay=15, starting_delay=10, label="control"
        )
        p.set_outputs(o)
        return p

    def tune_p(fr):
        p = make_p(fr)

        # Set initialization parameters of the stage
        p["control"].set_delay(change_delay)

        stage_config = p["control"].get_config()
        generate_test_param_file("DELAY", stage_config)
        return p

    do_test(make_p, tune_p, frame_size)


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

    def make_p(fr):
        p = Pipeline(channels, frame_size=fr)
        o = p.stage(FirDirect, p.i, coeffs_path=filter_path)
        p.set_outputs(o)
        return p

    do_test(make_p, None, frame_size)
