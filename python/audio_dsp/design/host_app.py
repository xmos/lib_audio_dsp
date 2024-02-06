from pathlib import Path


HOST_APP = Path("xvf_host")
PROTOCOL = "usb"

def set_host_app(host_app, protocol="usb"):
    """
    Set the host_app and the protocol to use for control

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
    print("HOST_APP = ", HOST_APP)
    if not HOST_APP.is_file():
        raise RuntimeError(f"Host App file {str(HOST_APP)} doesn't exist")
    PROTOCOL = protocol
    if PROTOCOL != "usb":
        raise RuntimeError(f"Host control over {PROTOCOL} protocol not supported. Only usb protocol supported")

def get_host_app():
    """
    Get the host_app and the protocol to use for control

    Returns
    ----------
    host_app and control protocol to use
    """
    if not HOST_APP.is_file():
        raise RuntimeError(f"Invalid Host App file {HOST_APP}. Call set_host_app() to set")
    if PROTOCOL != "usb":
        raise RuntimeError(f"Invalid host control protocol. Call set_host_app() to set")
    return HOST_APP, PROTOCOL
