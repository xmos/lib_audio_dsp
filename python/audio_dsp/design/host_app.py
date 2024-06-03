# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Global host app management, to provide easy access to the host app."""

from pathlib import Path
import platform
import subprocess


HOST_APP = Path("dsp_host")
PROTOCOL = "usb"
PORT = None


class InvalidHostAppError(Exception):
    """Raised when there is a issue with the configured host app."""

    pass


def set_host_app(host_app, transport_protocol="usb"):
    """
    Set the host_app and the transport protocol to use for control.

    Raises
    ------
    InvalidHostAppError
        If an invalid host app or transport protocol is selected.

    Parameters
    ----------
    host_app : str
        Host app file
    transport_protocol : str
        Protocol to use for control. Supported transport protocols are usb and xscope
    """
    global HOST_APP
    global PROTOCOL
    HOST_APP = Path(host_app)
    if platform.system().lower() == "windows" and HOST_APP.suffix != ".exe":
        HOST_APP = HOST_APP.with_suffix(".exe")
    if not HOST_APP.is_file():
        raise InvalidHostAppError(f"Host App file {str(HOST_APP)} doesn't exist")
    PROTOCOL = transport_protocol
    if PROTOCOL != "usb" and transport_protocol != "xscope":
        raise InvalidHostAppError(
            f"Host control over {PROTOCOL} transport protocol is not supported. Only usb or xscope protocols are supported."
        )


def set_host_app_xscope_port(port_num):
    """
    Set the port number on which to communicate with the device when doing control over xscope.

    Raises
    ------
    InvalidHostAppError
        If the port is set before calling set_host_app() or if the port is set when transport protocol is not xscope

    Parameters
    ----------
    host_app : int
        Port number

    Returns
    -------
    The return value from the subprocess.run(). The caller can use this to check the returncode, stdout etc.
    """
    global PORT
    if not HOST_APP.is_file():
        raise InvalidHostAppError(f"Invalid Host App file {HOST_APP}. Call set_host_app() to set")
    if PROTOCOL != "xscope":
        raise InvalidHostAppError("Port is set only for xscope transport protocol")
    PORT = port_num


def send_control_cmd(instance_id, *args, verbose=False):
    """
    Send a control command from the host to the device.

    Raises
    ------
    InvalidHostAppError
        If set_host_app() hasn't been called to set the host app and transport protocol before calling this function.
        If the transport protocol is 'xscope', and port num has not been set by calling set_host_app_xscope_port() before calling this function

    Parameters
    ----------
    instance_id : int | str
        Instance id of the stage to which this command is sent
    *args : list[str]
        Command + arguments for this control command
    verbose : bool
        When set to true, print the full command that gets issued
    """
    if not HOST_APP.is_file():
        raise InvalidHostAppError(f"Invalid Host App file {HOST_APP}. Call set_host_app() to set")
    if PROTOCOL != "usb" and PROTOCOL != "xscope":
        raise InvalidHostAppError(
            "Invalid host control transport protocol. Call set_host_app() to set"
        )
    if PROTOCOL == "xscope" and PORT is None:
        raise InvalidHostAppError("Port not set when using xscope transport protocol")

    cmd_list = [
        HOST_APP,
        "--use",
        PROTOCOL,
        "--instance-id",
        str(instance_id),
        *[i for i in args],
    ]
    print(cmd_list)
    if PROTOCOL == "xscope":
        cmd_list.extend(["--port", str(PORT)])

    ret = subprocess.run(
        cmd_list,
        stdout=subprocess.PIPE,
    )

    if ret.returncode:
        print(f"Unable to connect to device using {HOST_APP}")
        return ret

    if verbose:
        print(*cmd_list)

    return ret
