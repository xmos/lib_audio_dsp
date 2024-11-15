import os
from filelock import FileLock


def xdist_safe_bin_write(sig_int, sig_path):
  # write a integer signal (sig_int) to sig_path in a multithread safe way
  # If sig_path already exists, it will only be overwritten if running without
  # xdist
  worker_id = os.environ.get("PYTEST_XDIST_WORKER")
  with FileLock(str(sig_path) + ".lock"):
    # only write the file if it doesn't exist, or we're running single threaded
    if not sig_path.is_file() or worker_id is None:
      sig_int.tofile(sig_path)