========
DSP HOST
========

DSP host is a host control reference application.
It can be used with products integrated with Audio DSP.

********
Building
********

Build with cmake from the lib_audio_dsp/host folder:

.. note::

    Windows, Linux and macOS x86_64 need the XTC tools enabled.
    For these platforms the drivers for the xSCOPE protocol are built.
    These drivers require a library included in the XTC tools.

- on Linux and Mac

.. code-block:: console

    # Enable the XTC tools, except for macOS arm64
    cmake -B build && cd build && make

- on Windows

.. code-block:: console

    # Enable the XTC tools
    # building with VS tools
    cmake -G Ninja -B build && cd build && ninja

.. note::

    Windows drivers can only be built with 32-bit tools

*****
Using
*****

In order to use the application you should have the following files in the same location:

- dsp_host(.exe)
- (lib)command_map.(so/dll/dylib)
- (lib)device_{protocol}.(so/dll/dylib)

.. note::

    - Linux dynamic libraries end with ``.so``
    - Apple dynamic libraries end with ``.dylib``
    - Windows dynamic libraries don't have ``lib`` prefix and end with ``.dll``

The application and the device drivers can be obtained by following the build instructions of this repo. Command map is a part of the firmware code and built separately.
To find out use cases and more information about the application use:

- on Linux and Mac

.. code-block:: console

    ./dsp_host --help

- on Windows

.. code-block:: console

    dsp_host.exe --help

*****************************************
Supported platforms and control protocols
*****************************************

.. note:: Only the platforms that support the

- Linux - x86_64
    - dsp_host
    - libdevice_usb.so
    - libdevice_xscope.so
- Mac - x86_64
    - dsp_host
    - libdevice_usb.dylib
    - libdevice_xscope.dylib
- Mac - arm64
    - dsp_host
    - libdevice_usb.dylib
- Windows - x86 (32-bit)
    - dsp_host.exe
    - device_usb.dll
    - device_xscope.dll

