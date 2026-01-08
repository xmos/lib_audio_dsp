# Copyright 2024-2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

"""Collection of tuning utilities for the pipeline."""

from audio_dsp.design.pipeline import Pipeline
from audio_dsp.tuning.transport import *
from typing import Literal
import numpy as np
import tabulate


def _validate_pipeline_checksum(pipeline: Pipeline, proto: TuningTransport):
    """
    Check if Python and device pipeline checksums match.
    Raise a runtime error if the checksums are not equal.
    Assumes that proto is an already-connected TuningTransport.

    Parameters
    ----------
    pipeline : Python pipeline for which to validate checksum against the device pipeline
    """
    assert pipeline.pipeline_stage is not None  # To stop ruff from complaining

    payload = CommandPayload(pipeline.pipeline_stage, "checksum", None)
    device_pipeline_checksum = proto.read(payload)
    equal = np.array_equal(
        np.array(device_pipeline_checksum),
        np.array(pipeline.pipeline_stage["checksum"]),
    )

    if equal is False:
        raise RuntimeError(
            (
                "Device pipeline mismatch; the pipeline on the connected device does not match "
                "this design. To resolve this, update the firmware on the connected device to use this updated pipeline.\n"
                f"\n\tExpected checksum: {pipeline.pipeline_stage['checksum']}\n\tGot {device_pipeline_checksum}"
            )
        )


def send_config_to_device(pipeline: Pipeline, protocol: TuningTransport):
    """
    Send the current config for all stages to the device.
    Make sure set_host_app() is called before calling this to set a valid host app.

    Parameters
    ----------
    pipeline : Pipeline
        A designed and optionally tuned pipeline
    protocol : TuningTransport
        An initialised subclass of TuningTransport to use for communicating with the device
    """
    with protocol as proto:
        _validate_pipeline_checksum(pipeline, proto)

        for stage in pipeline.stages:
            for command, value in stage.get_config().items():
                payload = CommandPayload(stage, command, value)
                proto.write(payload)


_ValidationMode = Literal["once"] | Literal["never"] | Literal["always"]


class ConfigSender:
    """
    A stateful config sender which has a memory of the last configs that were sent to
    the device and only sends configs which have changed.

    Example usage::

        from audio_dsp.tuning.transport.xscope import XScopeTransport
        from audio_dsp.tuning import ConfigSender
        from audio_dsp.design.pipeline import Pipeline

        p = Pipeline(generate_xscope_task=True)
        t = XScopeTransport()
        c = config_sender(t)
        # Send config to the device
        c(p)


    Parameters
    ----------
    protocol: TuningTransport
        An initialised subclass of TuningTransport to use for communicating with the
        device.
    validation_mode: str
        One of "always", "never", or "once", determines how ofter the pipeline checksum is validated.
    """

    def __init__(self, protocol: TuningTransport, validation_mode: _ValidationMode = "always"):
        self._protocol = protocol
        self._last_config = {}
        self._validation_mode = validation_mode
        if self._validation_mode not in {"once", "always", "never"}:
            raise ValueError(f"Invalid validation mode {validation_mode}")
        self._validated = False

    def __call__(self, pipeline: Pipeline) -> None:
        """
        Send updated pipeline parameters.

        Parameters
        ----------
        pipeline:
            The pipeline to send to the device.
        """
        commands = {}
        for stage in pipeline.stages:
            stage_index = stage.index
            for command, value in stage.get_config().items():
                c = (stage_index, command, value)
                commands[str(c)] = c

        with self._protocol as proto:
            if self._validation_mode == "always" or (
                self._validation_mode == "once" and not self._validated
            ):
                _validate_pipeline_checksum(pipeline, self._protocol)
                self._validated = True
            for command_str in (c for c in commands if c not in self._last_config):
                command = commands[command_str]
                payload = CommandPayload(pipeline.stages[command[0]], command[1], command[2])
                proto.write(payload)

        self._last_config = commands


def profile_pipeline(pipeline: Pipeline, protocol: TuningTransport):
    """
    Profiles the DSP threads that are a part of the pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        A designed and optionally tuned pipeline
    """
    with protocol as proto:
        _validate_pipeline_checksum(pipeline, proto)

        # print("Thread Index     Max Cycles")
        profile_info = []
        for thread in pipeline.threads:
            thread_fs = None
            thread_frame_size = None
            stages = thread.get_all_stages()
            for stg in stages:
                if stg.fs is not None:
                    thread_fs = stg.fs
                    thread_frame_size = stg.frame_size
                    break
            # Assuming that all stages in the thread have the same sampling freq and frame size
            if thread_fs is None:
                raise RuntimeError(
                    f"Could not find out the sampling frequency for thread index {thread.id}"
                )

            if thread_frame_size is None:
                raise RuntimeError(
                    f"Could not find out the frame size for thread index {thread.id}"
                )

            reference_timer_freq_hz = 100e6
            frame_time_s = float(thread_frame_size) / thread_fs
            ticks_per_sample_time_s = reference_timer_freq_hz * frame_time_s
            ticks_per_sample_time_s = ticks_per_sample_time_s

            # TODO Implement a generic way of reading all config from the stage
            payload = CommandPayload(thread.thread_stage, "max_cycles", None)
            cycles = proto.read(payload)

            percentage_used = (cycles / ticks_per_sample_time_s) * 100
            profile_info.append(
                [
                    thread.id,
                    round(ticks_per_sample_time_s, 2),
                    cycles,
                    round(percentage_used, 2),
                ]
            )
        print(
            tabulate.tabulate(
                profile_info,
                headers=[
                    "thread index",
                    "available time (ref timer ticks)",
                    "max ticks consumed",
                    "% consumed",
                ],
                tablefmt="pretty",
            )
        )
