app_fd_block_fir
---

This demonstrates the usage of a frequency domain FIR. It runs a 4096 tap
bandpass filter with a 256 sample latency. This requires around 3x less
compute than the equivalent block time domain filter, and around 6x less
compute than a single sample implementation (both VPU optimised).

To build the example, first generate the test filter `test_0` by running `make_test_filters.py`:

.. code-block:: console
    
    cd examples/app_fd_block_fir/src
    python make_test_filters.py

Then autogenerate the frequency domain filter with a frame advance of 256 samples:

.. code-block:: console
    
    python -m audio_dsp.dsp.fd_block_fir test_0.npy 256

Finally, build the example app:

.. code-block:: console
    cd ..
    cmake -G "Unix Makefiles" -B build
    cmake -C build
