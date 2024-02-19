# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""
Global host app management, to provide easy access to the host app.
"""
from pathlib import Path
import platform


HOST_APP = Path("xvf_host")
PROTOCOL = "usb"

class InvalidHostAppError(Exception):
    """
    Raised when there is a issue with the configured host app.
    """
    pass

def set_host_app(host_app, protocol="usb"):
    """
    Set the host_app and the protocol to use for control

    Raises
    ------
    InvalidHostAppError
        If an invalid host app or protocol is selected.

    Parameters
    ----------
    host_app : str
        Host app file
    protocol : str
        Protocol to use for control. Only supported protocol is 'usb'
    """
    global HOST_APP
    global PROTOCOL
    HOST_APP = Path(host_app)
    if platform.system().lower() == "windows" and HOST_APP.suffix != '.exe':
        HOST_APP = HOST_APP.with_suffix('.exe')
    if not HOST_APP.is_file():
        raise InvalidHostAppError(f"Host App file {str(HOST_APP)} doesn't exist")
    PROTOCOL = protocol
    if PROTOCOL != "usb":
        raise InvalidHostAppError(f"Host control over {PROTOCOL} protocol not supported. Only usb protocol supported")

def get_host_app():
    """
    Get the host_app and the protocol to use for control

    Raises
    ------
    InvalidHostAppError
        If executable for binary not set

    Returns
    ----------
    host_app and control protocol to use
    """
    if not HOST_APP.is_file():
        raise InvalidHostAppError(f"Invalid Host App file {HOST_APP}. Call set_host_app() to set")
    if PROTOCOL != "usb":
        raise InvalidHostAppError(f"Invalid host control protocol. Call set_host_app() to set")
    return HOST_APP, PROTOCOL
