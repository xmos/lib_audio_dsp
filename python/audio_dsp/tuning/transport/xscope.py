
# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from . import CommandPayload, TuningTransport

class XScopeTransport(TuningTransport):
    """Manages all methods required to communicate tuning over xscope."""

    def __init__(self) -> None:
        super().__init__()

    def connect(self):
        return super().connect()
    
    def write(self, payload: CommandPayload):
        return super().write(payload)
    
    def read(self):
        return super().read()
    
    def disconnect(self):
        return super().disconnect()