# Copyright 2024-2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

"""This module implements the XScopeTransport class for managing DSP tuning
communication over xscope, including endpoint handling and command type mappings.
"""

from . import (
    CommandPayload,
    TuningTransport,
    DeviceConnectionError,
    MultiValType,
)
from .xscope_endpoint import Endpoint, QueueConsumer


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

        # Target schema is instance_id, cmd_id, payload_len, payload
        command_id = payload.cmd_id | (0x80 if read_cmd else 0x00)
        n_bytes, values = payload.to_bytes()
        payload_bytes = bytes([payload.stage_index, command_id, n_bytes])
        if values is not None:
            payload_bytes += values

        # print(f"sent: {payload_bytes}")
        return self.ep.publish(payload_bytes)

    def read(self, payload: CommandPayload) -> MultiValType:
        """
        Send a read command to the device over xscope, and then wait for the reply.

        Parameters
        ----------
        payload : CommandPayload
            The command to write.

        Returns
        -------
        ValType | tuple[ValType, ...]
            Tuple of data received from the device. The device sends raw bytes;
            this function automatically casts the received data to the type
            specified in payload.cmd_type.
        """
        self.write(payload, read_cmd=True)

        data = self.read_queue.next()
        # We know that for this specific application, data will always be bytes
        assert isinstance(data, bytes)

        return payload.from_bytes(data).values

    def disconnect(self):
        """Shut down the connection to the currently connected xscope server."""
        self.ep.disconnect()
        self.connected = False
