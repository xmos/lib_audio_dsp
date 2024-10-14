# Copyright 2024 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Python bindings for Tools-supplied xscope_endpoint.dll/.so."""

import os, ctypes, platform, sys, time, queue
from collections import defaultdict
import ctypes.util
import pathlib
import typing
import numpy as np

"""
 Function prototypes to match the c functions defined in xscope_endpoint.h
"""
PRINT_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_ulonglong,  # timestamp
    ctypes.c_uint,  # length
    ctypes.c_char_p,
)  # data

RECORD_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_uint,  # id
    ctypes.c_ulonglong,  # timestamp
    ctypes.c_uint,  # length
    ctypes.c_ulonglong,  # dataval
    ctypes.POINTER(ctypes.c_char),
)  # databytes

REGISTER_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_uint,  # id
    ctypes.c_uint,  # type
    ctypes.c_uint,  # r
    ctypes.c_uint,  # g
    ctypes.c_uint,  # b
    ctypes.c_char_p,  # name
    ctypes.c_char_p,  # unit
    ctypes.c_uint,  # data_type
    ctypes.c_char_p,
)  # data_name

STATS_CALLBACK = ctypes.CFUNCTYPE(
    None,
    ctypes.c_uint,  # id
    ctypes.c_ulonglong,
)  # average


ProbeInfo = typing.NamedTuple(
    "ProbeInfo",
    [
        ("type", int),
        ("r", int),
        ("g", int),
        ("b", int),
        ("name", str),
        ("unit", str),
        ("data_type", int),
        ("data_name", str),
    ],
)

ConsumerPrototype = typing.Callable[[int, str, int, int, bytes], None]


class Endpoint:
    """Python xSCOPE endpoint wrapper.

    Example:


        def test_callback(timestamp, probe_name, length, data_val, data_bytes):
            print(
                "{} {} {} {} {}".format(timestamp, probe_name, length, data_val, data_bytes)
            )

        ep = Endpoint()

        try:
            if ep.connect('localhost', '12345'):
                print "Failed to connect"
            else:
                ep.consume(callback, 'my_probe")
                while(True):
                    pass

        except KeyboardInterrupt:
            ep.disconnect()
    """

    def __init__(self):
        self._probe_info: dict[int, ProbeInfo] = {}  # probe id to probe info lookup.
        # probe_info includes name, units, data type, etc...
        self._consumers: typing.DefaultDict[str, set[ConsumerPrototype]] = defaultdict(
            set
        )  # probe name -> callbacks lookup
        # NOTE: The consumers must be looked up by name and not id because
        #      they can be specified before the probe_info is defined

        tool_path = os.environ.get("XMOS_TOOL_PATH")
        assert tool_path is not None
        tool_path = pathlib.Path(tool_path)

        ps = platform.system()
        if ps == "Windows":
            lib_path = tool_path / "lib" / "xscope_endpoint.dll"
        else:  # Darwin (aka MacOS) or Linux
            lib_path = tool_path / "lib" / "xscope_endpoint.so"
        self.lib_xscope = ctypes.CDLL(str(lib_path))

        # create callbacks
        print_cb = self._print_callback_func()
        self.lib_xscope.xscope_ep_set_print_cb(print_cb)

        record_cb = self._record_callback_func()
        self.lib_xscope.xscope_ep_set_record_cb(record_cb)

        register_cb = self._register_callback_func()
        self.lib_xscope.xscope_ep_set_register_cb(register_cb)

        stats_cb = self._stats_callback_func()
        self.lib_xscope.xscope_ep_set_stats_cb(stats_cb)

    def _print_callback_func(self):
        def func(timestamp: int, length: int, data: bytes):
            self.on_print(timestamp, data[0:length])

        return PRINT_CALLBACK(func)

    def _register_callback_func(self):
        def func(
            id_: int,
            type_: int,
            r: int,
            g: int,
            b: int,
            name: bytes,
            unit: bytes,
            data_type: int,
            data_name: bytes,
        ):
            pi = ProbeInfo(
                type_,
                r,
                g,
                b,
                name.decode(),
                unit.decode(),
                data_type,
                data_name.decode(),
            )
            self._probe_info[id_] = pi
            self.on_register(id_, type_, name, unit, data_type)

        return REGISTER_CALLBACK(func)

    def _record_callback_func(self):
        def func(id_: int, timestamp: int, length: int, data_val: int, data_bytes: bytes):
            self.on_record(id_, timestamp, length, data_val, data_bytes)

        return RECORD_CALLBACK(func)

    def _stats_callback_func(self):
        def func(id_: int, average: int):
            # TODO
            pass

        return STATS_CALLBACK(func)

    def on_print(self, timestamp: int, data: bytes):
        """XScope printf handler.
        Override this to method to implement your own printing or to silence the printout.
        """
        print("DEV: " + data.decode().rstrip())

    def on_register(self, id_: int, type_: int, name: bytes, unit: bytes, data_type: int):
        """Server probe registration handler.
        Override this to method to implement your own registration or to silence the printout.
        """
        print(f"Probe registered: {id_=}, {type_=}, {name=}, {unit=}, {data_type=}")

    def on_record(self, id_: int, timestamp: int, length: int, data_val: int, data_bytes: bytes):
        """Server record handler.  Will dispatch to probe consumer callback.
        Override this to method to implement your own dispatcher.  However,
        that should rarely be necessary.
        """
        # For some reason negative values keep getting interpreted as UINT64
        # which messes everything up. This seems to fix it.
        data_val = int(np.int32(np.uint32(data_val & 0xFFFFFFFF)))

        def notify_consumers(consumers: set[ConsumerPrototype], probe_name: str):
            for cb in consumers:
                cb(timestamp, probe_name, length, data_val, data_bytes)

        probe_info = self._probe_info[id_]
        probe_name = probe_info.name

        if probe_name in self._consumers:
            notify_consumers(self._consumers[probe_name], probe_name)
        if "*" in self._consumers:
            notify_consumers(self._consumers["*"], probe_name)

    def connect(self, hostname: str = "localhost", port: str = "12345") -> int:
        """Connect to xSCOPE server.

        Args:
            hostname (str): Hostname of running xSCOPE server.
            port (str): Port number of running xSCOPE server.

        Returns
        -------
            0 for success
            1 for failure
        """
        return self.lib_xscope.xscope_ep_connect(hostname.encode(), port.encode())

    def disconnect(self):
        """Disconnect from xSCOPE server."""
        self.lib_xscope.xscope_ep_disconnect()

    def consume(self, callback: ConsumerPrototype, probe_name: str | None = None):
        """Consume a probe by name.

        Note:
            Set probe=None to consume all probes.

        Args:
            callback (callable): Callback function with signature (timestamp, probe_name, length, data_val, data_bytes).
            probe_name (str): Probe name
        """
        probe_name = "*" if probe_name is None else probe_name
        self._consumers[probe_name].add(callback)

    def publish(self, data: bytes) -> int:
        """Publish message to endpoint.

        The length of data must be <= 255.

        Args:
            data: Bytes to publish.

        Returns
        -------
          0 for success
          1 for failure
        """
        return self.lib_xscope.xscope_ep_request_upload(
            ctypes.c_uint(len(data) + 1), ctypes.c_char_p(data)
        )


class QueueConsumer:
    """
    Attaches to a specified Endpoint class and probe. Automatically consumes all
    data from the probe and places it in a queue. The queue may then be queried
    with the .next() method.

    Parameters
    ----------
    ep : Endpoint
        Endpoint object to which to attach this consumer

    probe : str
        Name of the probe for which to queue data. Set to None to consume all probes.

    probe_timeout : float
    """

    def __init__(
        self,
        ep: Endpoint,
        probe: str,
        /,
        probe_timeout: float = 10.0,
        mode: typing.Literal["values", "bytes"] = "bytes",
    ):
        self.ep = ep
        self.probe_timeout = probe_timeout
        self.ep.consume(self._consume, probe)
        self.queue = queue.Queue()
        self.mode = mode

    def _consume(
        self,
        timestamp: int,
        probe_name: str,
        length: int,
        data_val: int,
        data_bytes: bytes,
    ):
        if self.mode == "values":
            self.queue.put(data_val)
        elif self.mode == "bytes":
            self.queue.put(data_bytes[:length])

    def next(self, count: int = 1) -> int | bytes | list[int] | list[bytes]:
        """
        Get the next item from the queue.

        Parameters
        ----------
        count : int
            Number of items from the queue to return. Defaults to 1.

        Raises
        ------
        Queue.Empty
            The probes timed out when trying to get values from xscope.
        """
        # At least in some scenarios, ctrl-c fails to interrupt the script and
        # just hangs forever if it's blocking on queue.get(). The timeout is
        # currently serving as a watchdog so that the pytest process doesn't need
        # to be killed through extraordinary means.
        if count == 1:
            return self.queue.get(timeout=self.probe_timeout)

        r = []
        for _ in range(count):
            r.append(self.queue.get(timeout=self.probe_timeout))
        return r


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Python xSCOPE")
    parser.add_argument("--host", default="localhost", help="Hostname")
    parser.add_argument("--port", default="12345", help="Port")
    parser.add_argument(
        "-c",
        "--consume",
        nargs="?",
        action="append",
        default=[],
        help="Probe names to consume (omit to consume all)",
    )
    parser.add_argument("-p", "--publish", default=None, help="Message to publish")
    args = parser.parse_args()

    def test_callback(timestamp, probe_name, length, data_val, data_bytes):
        """Prints received values."""
        print("{} {} {} {} {}".format(timestamp, probe_name, length, data_val, data_bytes))

    ep = Endpoint()
    try:
        if ep.connect(args.host, args.port):
            print("Failed to connect")
            sys.exit(1)

        if args.publish:
            ep.publish(args.publish)

        if args.consume:
            for probe in args.consume:
                ep.consume(test_callback, probe)
        else:
            ep.consume(test_callback)

        while True:
            # Release the CPU
            time.sleep(1)
    except KeyboardInterrupt:
        ep.disconnect()
