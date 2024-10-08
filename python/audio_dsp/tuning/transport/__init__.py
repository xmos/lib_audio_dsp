# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from __future__ import annotations

import abc
import collections
import contextlib

CommandPayload = collections.namedtuple(
    "CommandPayload",
    [
        ("index", int | None),
        ("command", str),
        ("value", str | list[str] | tuple[str, ...]),
    ],
)


class DeviceConnectionError(Exception):
    """Raised when the host app cannot connect to the device."""

    pass


class TuningTransport(abc.ABC, contextlib.AbstractContextManager):
    """Base class for different transport media for tuning commands."""

    def __enter__(self):
        return self.connect()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: contextlib.TracebackType | None,
    ) -> bool | None:
        return self.disconnect()

    @abc.abstractmethod
    def connect(self) -> TuningTransport:
        raise NotImplementedError

    @abc.abstractmethod
    def write(self, payload: CommandPayload):
        raise NotImplementedError

    @abc.abstractmethod
    def read(self):
        raise NotImplementedError

    @abc.abstractmethod
    def disconnect(self):
        raise NotImplementedError


from .xscope import XScopeTransport
