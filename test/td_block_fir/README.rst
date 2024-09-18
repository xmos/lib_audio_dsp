
To build:
~~~~~~~~~

This builds under ``xcommon_cmake`` so you'll need to have that repo somewhere handy
Build system will rely on the environment variable ``XMOS_CMAKE_PATH`` pointing to your copy of ``xcommon_cmake``.

On native windows you can do

.. code-block:: console

  cmake -G "Unix Makefiles" -B build
  xmake -C build

it will fetch lib_xcore_math on the same level as the ``dsp_ultra`` as checkout a branch with ``xcommon_cmake`` support,
we're going back to the old sandbox structure here.

For some reason on wsl it does not see the ``XMOS_CMAKE_PATH`` variable unless you give to the command, so you'll need to:

.. code-block:: console

  XMOS_CMAKE_PATH=~/your_path cmake -B build
  XMOS_CMAKE_PATH=~/your_path xmake -C build

After doing this ``bin/`` folder will have your binary
