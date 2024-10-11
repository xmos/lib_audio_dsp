# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

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
    """Raised when the host app cannot connect to the device."""

    pass


class TuningTransport(contextlib.AbstractContextManager["TuningTransport"], abc.ABC):
    """Base class for different transport media for tuning commands."""

    def __enter__(self) -> "TuningTransport":
        return self.connect()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool | None:
        return self.disconnect()

    @abc.abstractmethod
    def connect(self) -> "TuningTransport":
        raise NotImplementedError
        return self

    @abc.abstractmethod
    def write(self, payload: CommandPayload) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def read(self, payload: CommandPayload) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def disconnect(self):
        raise NotImplementedError


from .xscope import XScopeTransport
