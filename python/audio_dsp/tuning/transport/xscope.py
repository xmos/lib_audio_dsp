# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from __future__ import annotations
from . import CommandPayload, TuningTransport, DeviceConnectionError
from .xscope_endpoint import Endpoint


class XScopeTransport(TuningTransport):
    """Manages all methods required to communicate tuning over xscope."""

    def __init__(self, hostname: str = "localhost", port: str = "12345") -> None:
        self.ep = Endpoint()
        self.hostname = hostname
        self.port = port
        self.connected = False

    def connect(self) -> XScopeTransport:
        if not self.connected:
            ret = self.ep.connect(self.hostname, self.port)
            if ret == 0:
                self.connected = True
            else:
                raise DeviceConnectionError
        return self

    def write(self, payload: CommandPayload):
        if not self.connected:
            raise DeviceConnectionError
        self.ep.publish()

    def read(self):
        return super().read()

    def disconnect(self):
        self.ep.disconnect()
        self.connected = False
