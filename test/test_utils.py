# Copyright 2024-2025 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import os
from filelock import FileLock
import numpy as np
from audio_dsp.dsp.utils import Q_max
from audio_dsp.dsp.generic import Q_SIG

def float_to_qxx(arr_float, q = Q_SIG, dtype = np.int32):
  arr_int32 = np.clip((np.array(arr_float) * Q_max(q)), np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
  return arr_int32


def qxx_to_float(arr_int, q = Q_SIG):
  arr_float = np.array(arr_int).astype(np.float64) / Q_max(q)
  return arr_float

def q_convert_flt(arr_float, old_q, new_q, dtype = np.int32):
    arr_int32 = np.clip((np.array(arr_float) * Q_max(old_q)), np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    arr_int32 <<= (new_q - old_q)
    out_flt = arr_int32 / float(Q_max(new_q))
    return out_flt

def xdist_safe_bin_write(sig_int, sig_path):
  # write a integer signal (sig_int) to sig_path in a multithread safe way
  # If sig_path already exists, it will only be overwritten if running without
  # xdist
  worker_id = os.environ.get("PYTEST_XDIST_WORKER")
  with FileLock(str(sig_path) + ".lock"):
    # only write the file if it doesn't exist, or we're running single threaded
    if not sig_path.is_file() or worker_id is None:
      sig_int.tofile(sig_path)

def assert_allclose(actual, desired, rtol=1e-07, atol=0):
  np.testing.assert_allclose(actual[desired!=0], desired[desired!=0], rtol=rtol, atol=atol)