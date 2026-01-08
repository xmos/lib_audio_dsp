# Copyright 2024-2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""
The XMOS audio DSP Python library.

For generating DSP code for the xcore and simulating on a host PC.
"""

from importlib import metadata as _metadata
from packaging.version import Version as _Version
from functools import wraps
import warnings

__version__ = _metadata.version("audio_dsp")


class _ShouldHaveBeenDeletedException(Exception):
    pass


def _deprecated(since, removed_in, reason):
    """Identify functions as deprecated.

    Example
    -------

        from audio_dsp import _deprecated

        @_deprecated("1.0.0", "2.0.0", "Use X instead")
        def deprecated_x():
            ...
    """
    if _Version(__version__) >= _Version(removed_in):
        # prevent us from forgetting to remove deprecated stuff
        raise _ShouldHaveBeenDeletedException("Delete unsupported APIs")

    def decorator(func):
        # prepend the deprecation warning to the doc string. Adding it to the end
        # seems to cause some issues with numpydoc thinking that there is a parameter
        # named `deprecated`
        func.__doc__ = f"""
        .. deprecated:: {since}
           {func.__name__} will be removed in {removed_in}. {reason}

        """ + (func.__doc__ or "")

        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated and will be removed in {removed_in}. {reason}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
