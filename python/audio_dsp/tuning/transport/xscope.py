# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

"""This module implements the XScopeTransport class for managing DSP tuning
communication over xscope, including endpoint handling and command type mappings.
"""

from . import (
    CommandPayload,
    TuningTransport,
    DeviceConnectionError,
    DevicePayloadError,
    ValType,
    MultiValType,
)
from .xscope_endpoint import Endpoint, QueueConsumer
import struct
import typing
import numpy as np

CommandType = typing.NamedTuple("CommandType", [("type_name", str), ("type_size", int)])


class SilentEndpoint(Endpoint):
    """Subclass of Endpoint which silences the on_register callback.
    Consequently, this subclass does not generate any print() statements
    in normal operation.
    """

    def on_register(self, id_, type_, name, unit, data_type):
        """Handle server probe registration events. In this case, do nothing."""
        return


class XScopeTransport(TuningTransport):
    """
    Manages all methods required to communicate tuning over xscope.

    Parameters
    ----------
    hostname : str
        Hostname of the xscope server to which to attempt a connection.
        Defaults to 'localhost'.
    port : str
        Port of the xscope server to which to attempt a connection.
        Defaults to '12345'.
    probe_name : str
        Name of the xscope probe over which to receive data from the device.
        Defaults to 'ADSP'.
    """

    def __init__(
        self, hostname: str = "localhost", port: str = "12345", probe_name="ADSP"
    ) -> None:
        self.ep = SilentEndpoint()
        self.hostname = hostname
        self.port = port
        self.connected = False
        self.read_queue = QueueConsumer(self.ep, probe_name)
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

    def _transform_int(self, value: int, target_type: CommandType) -> bytes:
        if target_type.type_name in ("float", "float_s32_t"):
            raise DevicePayloadError
        return value.to_bytes(target_type.type_size, "big")

    def _transform_float(self, value: float, target_type: CommandType) -> bytes:
        if target_type.type_name not in ("float", "float_s32_t"):
            raise DevicePayloadError
        return bytes(struct.pack("!f", value))

    def _transform_npint(self, value: np.integer, target_type: CommandType) -> bytes:
        if target_type.type_name in ("float", "float_s32_t"):
            raise DevicePayloadError
        return value.tobytes()

    def _transform_single_value(self, value: ValType, target_type: CommandType) -> bytes:
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
            raise DevicePayloadError
        if values is None:
            return None

        target_type = CommandType(cmd_type, self.cmd_types_byte_lengths[cmd_type])

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
            raise DevicePayloadError

    def connect(self) -> "XScopeTransport":
        """
        Make a connection to a running xscope server.

        Returns
        -------
        self : XScopeTransport
            If this function returns, this object is guaranteed to be connected
            to a running xscope server

        Raises
        ------
        DeviceConnectionError
            If connection to the xscope server at {self.hostname}:{self.port} fails.
        """
        if not self.connected:
            ret = self.ep.connect(self.hostname, self.port)
            if ret == 0:
                self.connected = True
            else:
                raise DeviceConnectionError
        return self

    def write(self, payload: CommandPayload, read_cmd: bool = False) -> int:
        """
        Assemble a valid packet of bytes to send to the device via the connected
        xscope server, and then send them. Sets the top bit of the command ID if
        this is sending a read command.

        Parameters
        ----------
        payload : CommandPayload
            The command to write.
        read_cmd : bool
            Whether the command to be sent is a read command (True) or a write command (False).

        Returns
        -------
        int
            Return code: 0 if success, 1 if failure

        Raises
        ------
        DeviceConnectionError
            If this instance is not connected to a device. Call .connect().

        DevicePayloadError
            If the stated size of the payload in payload.size is not equal to
            the number of bytes that the value in payload.value is represented
            by when cast to the type in payload.cmd_type.
        """
        if not self.connected:
            raise DeviceConnectionError

        # Target schema is "ADSP", instance_id, cmd_id, payload_len, payload.
        # Start with the header
        payload_bytes = b"ADSP"
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
                raise DevicePayloadError
            payload_bytes += transformed_values
        # print(f"sent: {payload_bytes}")
        return self.ep.publish(payload_bytes)

    def read(self, payload: CommandPayload) -> tuple[ValType, ...]:
        """
        Send a read command to the device over xscope, and then wait for the reply.

        Parameters
        ----------
        payload : CommandPayload
            The command to write.

        Returns
        -------
        tuple[ValType, ...]
            Tuple of data received from the device. The device sends raw bytes;
            this function automatically casts the received data to the type
            specified in payload.cmd_type.
        """
        self.write(payload, read_cmd=True)

        data = self.read_queue.next()
        assert isinstance(data, bytes)

        # Device returns raw bytes. Cast to a tuple of whatever return value we need
        struct_code = f"{payload.size}{self.cmd_types_struct_map[payload.cmd_type]}"
        ret = struct.unpack(struct_code, data)

        if len(ret) == 1:
            ret = ret[0]
        return ret

    def disconnect(self):
        """Shut down the connection to the currently connected xscope server."""
        self.ep.disconnect()
        self.connected = False
