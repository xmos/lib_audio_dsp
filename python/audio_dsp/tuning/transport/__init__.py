# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from abc import ABC, abstractmethod
from collections import namedtuple

CommandPayload = namedtuple("CommandPayload", [("index", int | None), ("command", str), ("value", str | list[str] | tuple[str, ...])])

class DeviceConnectionError(Exception):
    """Raised when the host app cannot connect to the device."""

    pass

class TuningTransport(ABC):
    """Base class for different transport media for tuning commands."""

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def write(self, payload: CommandPayload):
        pass

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

from .xscope import XScopeTransport