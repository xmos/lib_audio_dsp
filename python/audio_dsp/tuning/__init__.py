# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

"""Collection of tuning utilities for the pipeline"""

from audio_dsp.design.pipeline import Pipeline
from audio_dsp.tuning.transport import *
import tabulate


class TuningInterface():

    # List of protocols currently supported by this library natively
    lib_protocols : dict[str, type[TuningTransport]] = {"xscope": XScopeTransport}

    def __init__(self, 
                 protocol : str ="xscope", 
                 custom_transport: TuningTransport | None = None) -> None:
        # If there isn't a custom transport specified, 
        # is the selected protocol string supported by the library?
        if custom_transport is None and protocol not in self.lib_protocols.keys():
            raise NotImplementedError(
                f"{protocol} is not a supported protocol.
                Supported protocols: {*self.lib_protocols.keys(),}"
                )
        # Use one of the library's inbuilt protocols 
        # if a custom one isn't specified
        self.protocol = custom_transport if custom_transport is not None else self.lib_protocols[protocol]


    def connect(self) -> None:
        self.protocol.connect()


    def send_config_to_device(self, pipeline: Pipeline):
        """
        Send the current config for all stages to the device.
        Make sure set_host_app() is called before calling this to set a valid host app.

        Parameters
        ----------
        pipeline : Pipeline
            A designed and optionally tuned pipeline
        """
        try:
            self.validate_pipeline_checksum(pipeline)
        except DeviceConnectionError:
            # Drop this exception, and print a warning
            print(
                "Unable to connect to device using host app. If using the Jupyter notebook, try to re-run all the cells."
            )
            return

        for stage in pipeline.stages:
            for command, value in stage.get_config().items():
                command = f"{stage.name}_{command}"
                if isinstance(value, list) or isinstance(value, tuple):
                    value = " ".join(str(v) for v in value)
                else:
                    value = str(value)
                payload = CommandPayload(stage.index, command, *value.split())
                self.protocol.write(payload)
                
    def validate_pipeline_checksum(self, pipeline: Pipeline):
        """
        Check if Python and device pipeline checksums match. Raise a runtime error if the checksums are not equal.
        The check is performed only if the host application can connect to the device.

        Parameters
        ----------
        pipeline : Python pipeline for which to validate checksum against the device pipeline
        """
        assert pipeline.pipeline_stage is not None  # To stop ruff from complaining

        payload = CommandPayload(pipeline.pipeline_stage.index, "pipeline_checksum", None)
        ret = self.protocol.write(payload)

        stdout = ret.stdout.decode().splitlines()
        device_pipeline_checksum = [int(x) for x in stdout]
        equal = np.array_equal(
            np.array(device_pipeline_checksum), np.array(pipeline.pipeline_stage["checksum"])
        )

        if equal is False:
            raise RuntimeError(
                f"Python pipeline checksum {pipeline.pipeline_stage['checksum']} does not match device pipeline checksum {device_pipeline_checksum}"
            )

    def profile_pipeline(self, pipeline: Pipeline):
        """
        Profiles the DSP threads that are a part of the pipeline.
        Make sure set_host_app() is called before calling this to set a valid host app.

        Parameters
        ----------
        pipeline : Pipeline
            A designed and optionally tuned pipeline
        """
        self.validate_pipeline_checksum(pipeline)

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
                raise RuntimeError(f"Could not find out the frame size for thread index {thread.id}")

            reference_timer_freq_hz = 100e6
            frame_time_s = float(thread_frame_size) / thread_fs
            ticks_per_sample_time_s = reference_timer_freq_hz * frame_time_s
            ticks_per_sample_time_s = ticks_per_sample_time_s

            # TODO Implement a generic way of reading all config from the stage
            payload = CommandPayload(thread.thread_stage.index, "dsp_thread_max_cycles", None)
            ret = self.protocol.write(payload)

            cycles = 0 # int(ret.stdout.splitlines()[0].decode("utf-8"))
            percentage_used = (cycles / ticks_per_sample_time_s) * 100
            profile_info.append(
                [thread.id, round(ticks_per_sample_time_s, 2), cycles, round(percentage_used, 2)]
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