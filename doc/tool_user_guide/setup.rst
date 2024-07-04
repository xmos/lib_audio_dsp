Setup
#####

In this section the requirements and the steps to run a basic pipeline are described. This document lists the necessary steps for both Windows and Linux/macOS.
This section uses the *app_simple_audio_dsp_integration* example found within this repository. The steps will be broadly similar for any user-created project.

.. note::

   Copying multiple lines into a console sometimes does not work as expected on Windows. Ensure that each line is copied and executed separately.

Hardware Requirements
=====================
- xcore.ai evaluation board (XK-EVK-XU316 or XK-316-AUDIO-MC-AB)
- xTag debugger and cable
- 2x Micro USB cable (one for power supply and one for the xTag)


Software Requirements
=====================

- `Graphviz <https://graphviz.org/download/#windows>`_: this software must installed and the ``dot`` executable must be on the system path.
- `XTC 15.2.1 <https://www.xmos.com/software-tools/>`_
- `Python 3.10 <https://www.python.org/downloads/>`_
- `CMake <https://cmake.org/download/>`_

Additionally, on Windows the following are required:
- `Visual Studio x86 native tools <https://visualstudio.microsoft.com/downloads/>`_
- `ninja-build <https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages#user-content-windows>`_
- `Zadig <https://zadig.akeo.ie/>`_

.. _all_steps:

Setup Steps
===========

.. note::

   All the steps below are executed from the sandbox folder created in the second step.

#. Prepare the development environment

   .. tab:: Windows

      On Windows:

      #. Open *x86 Native Tools Command Prompt*
      #. Activate the XTC environment:

         .. code-block:: console

            "C:\Program Files (x86)\XMOS\XTC\15.3.0\SetEnv"

         or similar

   .. tab:: Linux and macOS:

      On Linux and macOS:

      #. Open a terminal
      #. Activate the XTC environment using *SetEnv*

#. Create a sandbox folder with the command below:

   .. code-block:: console

      mkdir lib_audio_dsp_sandbox

#. Clone the library inside *lib_audio_dsp_sandbox*:

   .. code-block:: console

      git clone git@github.com:xmos/lib_audio_dsp.git

#. Get the sandbox inside *lib_audio_dsp_sandbox*. This step can take several minutes.

   .. tab:: Windows

      On Windows:

      .. code-block:: console

         cd lib_audio_dsp\examples\app_simple_audio_dsp_integration
         cmake -B build -G Ninja
         cd ..\..

   .. tab:: Linux and macOS

      On Linux and macOS:

      .. code-block:: console

         cd lib_audio_dsp/examples/app_simple_audio_dsp_integration
         cmake -B build
         cd ../..

#. Create a python virtualenv inside *lib_audio_dsp*.

   .. tab:: Windows

      On Windows:

      .. code-block:: console

         cd lib_audio_dsp
         python -m venv .venv
         .venv\Scripts\activate.bat
         pip install -Ur requirements.txt
         cd ..

   .. tab:: Linux and macOS

      On Linux or macOS:

      .. code-block:: console

         cd lib_audio_dsp
         python -m venv .venv
         source .venv/bin/activate
         pip install -Ur requirements.txt
         cd ..

#. Build and copy the host app files: these files are used to control the device from the host machine, and they must be placed in a specific location.

   .. tab:: Windows

      On Windows:

      .. code-block:: console

         cmake lib_audio_dsp\host\dsp_host -G Ninja -B lib_audio_dsp\host\dsp_host\build
         cmake --build lib_audio_dsp\host\dsp_host\build

         cmake lib_audio_dsp\host\host_cmd_map -B lib_audio_dsp\host\host_cmd_map\build -G Ninja
         cmake --build lib_audio_dsp\host\host_cmd_map\build

         robocopy lib_audio_dsp\host\dsp_host\build lib_audio_dsp\host\host_cmd_map\build *.exe *.dll

   .. tab:: Linux and macOS

      On Linux and macOS:

      .. code-block:: console

         cmake lib_audio_dsp/host/dsp_host -B lib_audio_dsp/host/dsp_host/build
         cmake --build lib_audio_dsp/host/dsp_host/build

         cmake lib_audio_dsp/host/host_cmd_map -B lib_audio_dsp/host/host_cmd_map/build
         cmake --build lib_audio_dsp/host/host_cmd_map/build

         cp -r lib_audio_dsp/host/dsp_host/build/* lib_audio_dsp/host/host_cmd_map/build/

#. If using Linux, :ref:`update the UDEV rules<udev_rules_linux>`.

#. Connect an XCORE-AI-EXPLORER using both USB ports

#. Open the notebook by running from *lib_audio_dsp_sandbox* the following command:

   .. code-block:: console

      jupyter notebook lib_audio_dsp/examples/app_simple_audio_dsp_integration/dsp_design.ipynb

   If a blank screen appears or nothing opens, then copy the link starting with "http://127.0.0.1/" from the terminal into the browser. The following page should open:

   .. figure:: images/jupyter_notebook_top_level.png
      :width: 100%

      Top-level page of the Jupyter Notebook

#. Run all the cells from the browser. From the menu at the top of the page click *Run -> Run all cells*:

   .. figure:: images/jupyter_notebook_run_tests.png
      :width: 100%

      Run menu of the Jupyter Notebook

   This creates the pipeline, builds the app and runs it on the device using xrun. Wait for all the cells to finish

   .. note::

      If running on Windows, you need to :ref:`install the libusb driver<libusb_windows>` after the first run.

   Any configuration or compilation errors will be displayed in the notebook in the *Build and Run* cell, as in the example below:

   .. figure:: images/config_error.png
      :width: 100%

      Configuration error of the Jupyter Notebook

   If there is any connection error, as in the example below:

   .. figure:: images/connect_error.png
      :width: 100%

      Connection error of the Jupyter Notebook

   please check the following:

      * Is the device connected to the host?
      * Are the host app files all present in the location reported in the error?
      * If using Windows, have the instructions to :ref:`install the libusb driver<libusb_windows>` been followed?
      * If using Linux, have the instructions to :ref:`update the UDEV rules<udev_rules_linux>` been followed?

#. Update and run *Pipeline design stage* to add the desired audio processing blocks. A diagram will be generated showing the pipeline IO mapping:

   * inputs 0 to 3 map respectively to USB OUT left, USB OUT right, line IN left and line IN right
   * outputs 0 to 3 map respectively to USB IN left, USB IN right, line OUT left and line OUT right.

   A simple pipeline example is shown in :numref:`pipeline_diagram`:

      .. _pipeline_diagram:

      .. figure:: images/pipeline_diagram.png
         :width: 100%

         Diagram of a simple audio pipeline

   See the top of the notebook for more information about this stage.


#. Update and run the *Tuning Stage* cell to change the parameters using the host app. See the top of the notebook for more information about this stage.

.. _libusb_windows:

Installing the libusb driver on Windows
=======================================

The first time the device is used on Windows, the libusb driver must be installed. This is done using the third-party tool *Zadig*.

**These steps are only required once and they must be executed while the firmware is running on the device**.

#. Open *Zadig* and select *XMOS Control (Interface 3)* from the list of devices.
   If the device is not present, ensure *Options -> List All Devices* is checked.

#. Select *libusb-win32* from the list of drivers in the right hand spin box.

#. Click the *Install Driver* button and wait for the installation to complete.

   .. figure:: images/zadig_install_control.png
      :width: 100%

      Selecting the *libusb-win32* driver in Zadig for the Control Interface

.. _udev_rules_linux:

Updating the UDEV rules on Linux
================================

The first time the device is used on Linux, the new UDEV rules must be installed.

#. Run the commands below to add the UDEV rules:

   .. code-block:: console

      UDEV_RULES_FILE=/etc/udev/rules.d/99-xmos-dsp.rules
      echo "SUBSYSTEM!=\"usb|usb_device\", GOTO=\"xmos_dsp_rules_end\"" | \
      sudo tee $UDEV_RULES_FILE
      echo "ACTION!=\"add\", GOTO=\"xmos_dsp_rules_end\"" | \
      sudo tee -a $UDEV_RULES_FILE
      echo "ATTRS{idVendor}==\"20b1\", ATTRS{idProduct}==\"5000\", MODE=\"0666\"," \
      "SYMLINK+=\"xCORE.ai-MC-%n\"" | sudo tee -a $UDEV_RULES_FILE

#. Reload the UDEV rules:

   .. code-block:: console

      sudo udevadm control --reload-rules
      sudo udevadm trigger

Running audio through the device
================================

Once running the device will enumerate as "XMOS xCORE (UAC2)"; it appears as a microphone
and a speaker and can be used as such to play signals from a PC.

On Windows it is possible to encounter
issues if the driver has picked different sample rates for the speaker and microphone. To fix this, follow the steps below:

#. From the start menu go to *Sound Settings -> Sound Control Panel*

#. Double click on the XMOS device in the *Playback* tab. In the new window go to the *Advanced* tab, and select 48000 Hz as sample rate

#. Double click on the XMOS device in the *Recording* tab. In the new window go to the *Advanced* tab, and select 48000 Hz as sample rate

Running an application after the first installation
===================================================

If running the application after the initial configuration, the following steps are required:

#. Configure the settings below, using the instructions in the :ref:`Setup Steps<all_steps>` section:

   * Enable the XTC tools: the installation can be tested by running the command ``xrun --version`` from the terminal. If the command is not found, the XTC tools are not installed correctly.
   * Set the path to the *xcommon_cmake* folder: this is checked by running the command ``echo %XMOS_CMAKE_PATH%`` on Windows, or ``echo $XMOS_CMAKE_PATH`` on Linux or macOS. The path should have been set.
   * Enable the Python Virtual Environment: this is checked by running the command ``echo %VIRTUAL_ENV%`` on Windows, or ``echo $VIRTUAL_ENV`` on Linux or macOS.  The path should have been set.
   * On Windows only, enable the VisualStudio (VS) tools: this can be checked by running the command ``cl`` from the terminal. If the command is not found, the VS tools are not installed correctly.

#. Open the notebook by running ``jupyter notebook lib_audio_dsp/examples/app_simple_audio_dsp_integration/dsp_design.ipynb`` from ``lib_audio_dsp_sandbox``, as described in :ref:`Setup Steps<all_steps>` section.
