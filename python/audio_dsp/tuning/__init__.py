# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

"""Collection of tuning utilities for the pipeline."""

from audio_dsp.design.pipeline import Pipeline
from audio_dsp.tuning.transport import *
from audio_dsp.design.stage import Stage
import numpy as np
import tabulate


class TuningInterface:

    # List of protocols currently supported by this library natively
    _lib_protocols: dict[str, type[TuningTransport]] = {"xscope": XScopeTransport}

    def __init__(
        self,
        protocol: str = "xscope",
        custom_transport: type[TuningTransport] | None = None,
    ) -> None:
        # If there isn't a custom transport specified,
        # is the selected protocol string supported by the library?
        if custom_transport is None and protocol not in self._lib_protocols.keys():
            raise NotImplementedError(
                (
                    f"{protocol} is not a supported protocol. "
                    f"Supported protocols: {*self._lib_protocols.keys(),}"
                )
            )
        # Use one of the library's inbuilt protocols
        # if a custom one isn't specified
        self._protocol = (
            custom_transport
            if custom_transport is not None
            else self._lib_protocols[protocol]
        )

    def _construct_payload(self, stage: Stage, command: str, value: MultiValType) -> CommandPayload:
        assert stage.index is not None
        assert stage.yaml_dict is not None

        stage_index = stage.index
        name = stage.name
        full_yaml: dict[str, dict] = stage.yaml_dict
        try:
            cmd_index = list(full_yaml['module'][name].keys()).index(command) + 1
            cmd_type = full_yaml['module'][name][command]['type']
            cmd_size = full_yaml['module'][name][command].get('size', 1)
        except ValueError as e:
            print(f"Command {command} not valid for stage {name}")
            raise e from None
        return CommandPayload(stage_index, cmd_index, value, cmd_size, cmd_type)


    def _validate_pipeline_checksum(self, pipeline: Pipeline, proto: TuningTransport):
        """
        Check if Python and device pipeline checksums match.
        Raise a runtime error if the checksums are not equal.
        Assumes that proto is an already-connected TuningTransport.

        Parameters
        ----------
        pipeline : Python pipeline for which to validate checksum against the device pipeline
        """
        assert pipeline.pipeline_stage is not None  # To stop ruff from complaining

        payload = self._construct_payload(pipeline.pipeline_stage, "checksum", None)
        device_pipeline_checksum = proto.read(payload)
        equal = np.array_equal(
            np.array(device_pipeline_checksum),
            np.array(pipeline.pipeline_stage["checksum"]),
        )

        if equal is False:
            raise RuntimeError(
                (
                    f"Python pipeline checksum {pipeline.pipeline_stage['checksum']} "
                    f"does not match device pipeline checksum {device_pipeline_checksum}"
                )
            )

    def send_config_to_device(self, pipeline: Pipeline):
        """
        Send the current config for all stages to the device.
        Make sure set_host_app() is called before calling this to set a valid host app.

        Parameters
        ----------
        pipeline : Pipeline
            A designed and optionally tuned pipeline
        """
        with self._protocol() as proto:
            self._validate_pipeline_checksum(pipeline, proto)

            for stage in pipeline.stages:
                for command, value in stage.get_config().items():
                    payload = self._construct_payload(stage, command, value)
                    proto.write(payload)

    def profile_pipeline(self, pipeline: Pipeline):
        """
        Profiles the DSP threads that are a part of the pipeline.

        Parameters
        ----------
        pipeline : Pipeline
            A designed and optionally tuned pipeline
        """
        with self._protocol() as proto:
            self._validate_pipeline_checksum(pipeline, proto)

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
                payload = self._construct_payload(thread.thread_stage, "max_cycles", None)
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
