# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

"""This module defines the base classes and types for DSP tuning transport mechanisms."""

import abc
import typing
import types
import contextlib
import numpy as np

ValType = int | float | str | np.integer
MultiValType = ValType | list[ValType] | tuple[ValType, ...] | None

CommandPayload = typing.NamedTuple(
    "CommandPayload",
    [
        ("index", int),
        ("command", int),
        ("value", MultiValType),
        ("size", int),
        ("cmd_type", str),
    ],
)


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
    def read(self, payload: CommandPayload) -> tuple[ValType, ...]:
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
