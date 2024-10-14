# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from __future__ import annotations
from . import (
    CommandPayload,
    TuningTransport,
    DeviceConnectionError,
    ValType,
    MultiValType,
)
from .xscope_endpoint import Endpoint, QueueConsumer
import struct
import numpy as np
from time import sleep


class SilentEndpoint(Endpoint):
    def on_register(self, id_, type_, name, unit, data_type):
        return


class XScopeTransport(TuningTransport):
    """Manages all methods required to communicate tuning over xscope."""

    def __init__(self, hostname: str = "localhost", port: str = "12345") -> None:
        self.ep = SilentEndpoint()
        self.hostname = hostname
        self.port = port
        self.connected = False
        self.read_queue = QueueConsumer(self.ep, "ADSP")
        self.cmd_types_byte_lengths = {
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
        self.cmd_types_struct_map = {
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

    def _transform_int(self, value: int, target_type: tuple[str, int]) -> bytes:
        type_name, type_size = target_type
        if type_name == "float" or type_name == "float_s32_t":
            raise TypeError
        return value.to_bytes(type_size, "big")

    def _transform_float(self, value: float, target_type: tuple[str, int]) -> bytes:
        type_name, _ = target_type
        if type_name != "float" and type_name != "float_s32_t":
            raise TypeError
        return bytes(struct.pack("!f", value))

    def _transform_npint(
        self, value: np.integer, target_type: tuple[str, int]
    ) -> bytes:
        type_name, _ = target_type
        if type_name == "float" or type_name == "float_s32_t":
            raise TypeError
        return value.tobytes()

    def _transform_single_value(
        self, value: ValType, target_type: tuple[str, int]
    ) -> bytes:
        if isinstance(value, int):
            return self._transform_int(value, target_type)
        elif isinstance(value, float):
            return self._transform_float(value, target_type)
        elif isinstance(value, np.integer):
            return self._transform_npint(value, target_type)
        else:  # string
            if "." in value:
                # treat strings as floats if they have a . in them.
                # there are error cases https://stackoverflow.com/a/20929881
                return self._transform_float(float(value), target_type)
            else:
                return self._transform_int(int(value), target_type)

    def _transform_values(self, values: MultiValType, cmd_type: str) -> bytes | None:
        if cmd_type not in self.cmd_types_byte_lengths:
            raise TypeError
        if values is None:
            return None

        target_type = (cmd_type, self.cmd_types_byte_lengths[cmd_type])

        if isinstance(values, (int, float, str, np.integer)):
            # Single argument
            return self._transform_single_value(values, target_type)
        elif isinstance(values, (list, tuple)):
            # Multiple arguments
            concat = bytes()
            for elem in values:
                concat += self._transform_single_value(elem, target_type)
            return concat
        else:
            # ???
            raise ValueError

    def connect(self) -> XScopeTransport:
        if not self.connected:
            ret = self.ep.connect(self.hostname, self.port)
            if ret == 0:
                self.connected = True
            else:
                raise DeviceConnectionError
        return self

    def write(self, payload: CommandPayload, read_cmd=False) -> int:
        if not self.connected:
            raise DeviceConnectionError

        # Target schema is "ADSP", instance_id, cmd_id, payload_len, payload.
        # Start with the header
        payload_bytes = b'ADSP'
        # Extract the instance_id, cmd_id, payload_len from payload
        command_id = payload.command | (0x80 if read_cmd else 0x00)
        command_size = payload.size * self.cmd_types_byte_lengths[payload.cmd_type]
        payload_bytes += bytes([payload.index, command_id, command_size])
        # Add on the transformed values
        transformed_values = self._transform_values(payload.value, payload.cmd_type)
        if transformed_values is not None:
            if len(transformed_values) != command_size:
                print(
                    f"Length error: {len(transformed_values)} != {command_size} for {payload.index}:{command_id} with value {payload.value}"
                )
                raise ValueError
            payload_bytes += transformed_values
        # print(f"sent: {payload_bytes}")
        return self.ep.publish(payload_bytes)

    def read(self, payload: CommandPayload) -> tuple[ValType, ...]:
        self.write(payload, read_cmd=True)

        data = self.read_queue.next()

        # Device returns raw bytes. Cast to a tuple of whatever return value we need
        struct_code = f"{payload.size}{self.cmd_types_struct_map[payload.cmd_type]}"
        ret = struct.unpack(struct_code, data)

        if len(ret) == 1:
            ret = ret[0]
        return ret

    def disconnect(self):
        self.ep.disconnect()
        self.connected = False
