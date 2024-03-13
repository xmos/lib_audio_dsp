# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Global host app management, to provide easy access to the host app."""

from pathlib import Path
import platform
import subprocess


HOST_APP = Path("xvf_host")
PROTOCOL = "usb"
PORT = None


class InvalidHostAppError(Exception):
    """Raised when there is a issue with the configured host app."""

    pass


def set_host_app(host_app, protocol="usb"):
    """
    Set the host_app and the protocol to use for control.

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
    if platform.system().lower() == "windows" and HOST_APP.suffix != ".exe":
        HOST_APP = HOST_APP.with_suffix(".exe")
    if not HOST_APP.is_file():
        raise InvalidHostAppError(f"Host App file {str(HOST_APP)} doesn't exist")
    PROTOCOL = protocol
    if PROTOCOL != "usb" and protocol != "xscope":
        raise InvalidHostAppError(
            f"Host control over {PROTOCOL} protocol not supported. Only usb or xscope protocol supported"
        )


def set_host_app_xscope_port(port_num):
    global PORT
    if not HOST_APP.is_file():
        raise InvalidHostAppError(f"Invalid Host App file {HOST_APP}. Call set_host_app() to set")
    if PROTOCOL != "xscope":
        raise InvalidHostAppError("Port is set only for xscope protocol")
    PORT = port_num


def send_host_cmd(instance_id, *args, verbose=False):
    if not HOST_APP.is_file():
        raise InvalidHostAppError(f"Invalid Host App file {HOST_APP}. Call set_host_app() to set")
    if PROTOCOL != "usb" and PROTOCOL != "xscope":
        raise InvalidHostAppError("Invalid host control protocol. Call set_host_app() to set")
    if PROTOCOL == "xscope" and PORT is None:
        raise InvalidHostAppError("Port not set when using xscope protocol")

    if PROTOCOL != "xscope":
        ret = subprocess.run(
            [HOST_APP, "--use", PROTOCOL, "--instance-id", str(instance_id), *[i for i in args]],
            stdout=subprocess.PIPE,
        )
    else:
        ret = subprocess.run(
            [
                HOST_APP,
                "--use",
                PROTOCOL,
                "--instance-id",
                str(instance_id),
                "--port",
                str(PORT),
                *[i for i in args],
            ],
            stdout=subprocess.PIPE,
        )
    if ret.returncode:
        print(f"Unable to connect to device using {HOST_APP}")
        return ret
    if verbose:
        print(HOST_APP, "--use", PROTOCOL, "--instance-id", str(instance_id), *[i for i in args])
    return ret
