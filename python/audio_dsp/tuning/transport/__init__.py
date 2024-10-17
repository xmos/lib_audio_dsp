# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

"""This module defines the base classes and types for DSP tuning transport mechanisms."""

import abc
import contextlib
import numpy as np
import struct
import types
from audio_dsp.design.stage import Stage

ValType = int | float | str | np.integer
MultiValType = ValType | list[ValType] | tuple[ValType, ...] | None


class CommandPayload:
    """Class for holding all relevant information regarding a command."""

    cmd_types_byte_lengths = {
        "uint8_t": 1,
        "int8_t": 1,
        "int16_t": 2,
        "int": 4,
        "int32_t": 4,
        "uint32_t": 4,
        "int32_t*": 4,
        "float": 4,
        "float_s32_t": 4,
    }
    cmd_types_struct_map = {
        "uint8_t": "B",
        "int8_t": "b",
        "int16_t": "h",
        "int": "i",
        "int32_t": "i",
        "uint32_t": "I",
        "int32_t*": "I",
        "float": "f",
        "float_s32_t": "f",
    }

    def __init__(self, stage: Stage, command: str, value: MultiValType) -> None:
        assert stage.index is not None
        assert stage.yaml_dict is not None

        stage_index = stage.index
        name = stage.name
        module_yaml: dict[str, dict] = stage.yaml_dict["module"][name]
        try:
            cmd_index = list(module_yaml.keys()).index(command) + 1
            cmd_type = module_yaml[command]["type"]
            cmd_n_values = module_yaml[command].get("size", 1)
        except ValueError as e:
            print(f"Command {command} not valid for stage {name}")
            raise e from None

        self.stage_index: int = stage_index
        self.values: MultiValType = value
        self.cmd_id: int = cmd_index
        self.cmd_n_values: int = cmd_n_values
        self.cmd_type: str = cmd_type
        self.stage: Stage = stage
        self.command: str = command

    def to_bytes(self) -> tuple[int, bytes | None]:
        """Convert this commands' values into a set of raw bytes."""
        retnum = 0
        retvals = None

        try:
            cmd_n_bytes = self.cmd_n_values * self.cmd_types_byte_lengths[self.cmd_type]
        except IndexError as e:
            print(f"Command type {self.cmd_type} size unknown, please add to class")
            raise e from None

        if self.values is not None:
            if isinstance(self.values, (int, float, str, np.integer)):
                # Single argument
                retvals = self._transform_single_value(self.values)
                retnum = len(retvals)
            elif isinstance(self.values, (list, tuple)):
                # Multiple arguments
                concat = bytes()
                for elem in self.values:
                    concat += self._transform_single_value(elem)
                retvals = concat
                retnum = len(retvals)
            else:
                # ???
                print(f"Transformation of values of type {type(self.values)} unknown.")
                raise DevicePayloadError

            if retnum != cmd_n_bytes:
                print(
                    f"Length error: {retnum} != {cmd_n_bytes} for {self.stage_index}:{self.cmd_id} with value {self.values}"
                )
                raise DevicePayloadError

        return (retnum, retvals)

    def _transform_single_value(self, value: ValType) -> bytes:
        if isinstance(value, int):
            return self._transform_int(value)
        elif isinstance(value, float):
            return self._transform_float(value)
        elif isinstance(value, np.integer):
            return self._transform_npint(value)
        else:  # string
            if "." in value:
                # treat strings as floats if they have a . in them.
                # there are error cases https://stackoverflow.com/a/20929881
                return self._transform_float(float(value))
            else:
                return self._transform_int(int(value))

    def _transform_int(self, value: int) -> bytes:
        if self.cmd_type in ("float", "float_s32_t"):
            raise DevicePayloadError
        return value.to_bytes(self.cmd_n_values, "big")

    def _transform_float(self, value: float) -> bytes:
        if self.cmd_type not in ("float", "float_s32_t"):
            raise DevicePayloadError
        return bytes(struct.pack("!f", value))

    def _transform_npint(self, value: np.integer) -> bytes:
        if self.cmd_type in ("float", "float_s32_t"):
            raise DevicePayloadError
        return value.tobytes()

    def from_bytes(self, data: bytes) -> "CommandPayload":
        """Convert bytes received from the device on issue of this command to
        bytes and return a valid CommandPayload object.
        """
        struct_code = f"{self.cmd_n_values}{self.cmd_types_struct_map[self.cmd_type]}"
        ret: MultiValType = struct.unpack(struct_code, data)

        if len(ret) == 1:
            ret = ret[0]
        return CommandPayload(self.stage, self.command, ret)


class DeviceConnectionError(Exception):
    """Raised when the tuning transport classes cannot connect to the device."""

    pass


class DevicePayloadError(Exception):
    """Raised when the payload specified to a transport class is malformed."""


class TuningTransport(contextlib.AbstractContextManager["TuningTransport"], abc.ABC):
    """Base class for different transport media for tuning commands."""

    def __enter__(self) -> "TuningTransport":
        """
        Call subclass' connect() method. This ensures that the subclass is
        instantiated correctly when used as a context manager.
        """
        return self.connect()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool | None:
        """
        Call the subclass' disconnect() method. Ensures that the subclass
        disconnects cleanly when used as a context manager, including when
        exceptions occur.
        """
        return self.disconnect()

    @abc.abstractmethod
    def connect(self) -> "TuningTransport":
        """
        Perform any required operations to set up a connection to the device
        and make the device ready to receive control commands.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def write(self, payload: CommandPayload) -> int:
        """Send a command to the device."""
        raise NotImplementedError

    @abc.abstractmethod
    def read(self, payload: CommandPayload) -> MultiValType:
        """
        Read data from the device. This is expected to perform a write operation
        with no payload to request the device make ready the requested data,
        then the device is expected to transmit the requested data back to the host.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def disconnect(self):
        """
        Perform any required operations to cleanly shut down the interface with
        the device.
        """
        raise NotImplementedError


from .xscope import XScopeTransport
