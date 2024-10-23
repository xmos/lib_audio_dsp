Setup
#####

This section describes the requirements and the steps to run a basic pipeline.
This document lists the necessary steps for both Windows and Linux/macOS.
This section uses the *app_simple_audio_dsp_integration* example found within this repository.
The steps will be broadly similar for any user-created project.

.. note::

   Copying multiple lines into a console sometimes does not work as expected on
   Windows. Ensure that each line is copied and executed separately.

Hardware Requirements
=====================
- xcore.ai evaluation board (XK-EVK-XU316 or XK-316-AUDIO-MC-AB)
- xTag debugger and cable
- 2x Micro USB cable (one for power supply and one for the xTag)


Software Requirements
=====================

- `Graphviz <https://graphviz.org/download/#windows>`_: this software must
  installed and the ``dot`` executable must be on the system path.
- `XTC 15.3.0 <https://www.xmos.com/software-tools/>`_
- `Python 3.10 <https://www.python.org/downloads/>`_
- `Jupyter 7.2.1 <https://jupyter.org/install>`_
- `CMake <https://cmake.org/download/>`_

Additionally, on Windows the following is required: 

- `ninja-build <https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages#user-content-windows>`_

.. _all_steps:

Setup Steps
===========

.. note::

   All the steps below are executed from the sandbox folder created in the
   second step.

#. Prepare the development environment

   .. tab:: Windows

      On Windows:

      #. Open the Command Prompt or other terminal application of choice
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

#. Get the sandbox inside *lib_audio_dsp_sandbox*. This step can take several
   minutes.

   .. tab:: Windows

      On Windows:

      .. code-block:: console

         cd lib_audio_dsp\examples\app_simple_audio_dsp_integration 
         cmake -B build -G Ninja cd ..\..\..\..

   .. tab:: Linux and macOS

      On Linux and macOS:

      .. code-block:: console

         cd lib_audio_dsp/examples/app_simple_audio_dsp_integration 
         cmake -B build 
         cd ../../../..

#. Create a requirements file inside *lib_audio_dsp_sandbox*.

   .. tab:: Windows

      On Windows:

      .. code-block:: console

         echo -e lib_audio_dsp/python > requirements.txt
         echo notebook >> requirements.txt

   .. tab:: Linux and macOS

      On Linux or macOS:

      .. code-block:: console

         echo "-e lib_audio_dsp/python" > requirements.txt
         echo notebook >> requirements.txt
         chmod 644 requirements.txt

#. Create a Python virtualenv inside *lib_audio_dsp_sandbox*.

   .. tab:: Windows

      On Windows:

      .. code-block:: console

         python -m venv .venv 
         .venv\Scripts\activate.bat 
         pip install -Ur requirements.txt 
         cd ..

   .. tab:: Linux and macOS

      On Linux or macOS:

      .. code-block:: console

         python -m venv .venv 
         source .venv/bin/activate 
         pip install -Ur requirements.txt 
         cd ..

#. Connect an XCORE-AI-EXPLORER using both USB ports

#. Open the notebook by running from *lib_audio_dsp_sandbox* the following
   command:

   .. code-block:: console

      jupyter notebook lib_audio_dsp/examples/app_simple_audio_dsp_integration/dsp_design.ipynb

   If a blank screen appears or nothing opens, then copy the link starting with
   "http://127.0.0.1/" from the terminal into the browser. The following page
   should open:

   .. figure:: ../images/jupyter_notebook_top_level.png
      :width: 25%

      Top-level page of the Jupyter Notebook

#. Run all the cells from the browser. From the menu at the top of the page
   click *Run -> Run all cells*:

   .. figure:: ../images/jupyter_notebook_run_tests.png
      :width: 100%

      Run menu of the Jupyter Notebook

   This creates the pipeline and builds the app. Wait for all the cells to
   finish

   Any configuration or compilation errors will be displayed in the notebook in
   the *Build and run* cell, as in the example below:

   .. figure:: ../images/config_error.png
      :width: 100%

      Run error of the Jupyter Notebook

#. Update and run *Pipeline design stage* to add the desired audio processing
   blocks. A diagram will be generated showing the pipeline IO mapping.

   A simple pipeline example is shown in :numref:`pipeline_diagram`:

      .. _pipeline_diagram:

      .. figure:: ../images/pipeline_diagram.png
         :width: 25%

         Diagram of a simple audio pipeline

   See the top of the notebook for more information about this stage.


#. Update and run the *Tuning Stage* cell to change the parameters before
   building. See the top of the notebook for more information about this stage.

Running a notebook after the first installation
===================================================

If running the notebook after the initial configuration, the following steps are
required:

#. Configure the settings below, using the instructions in the :ref:`Setup
   Steps<all_steps>` section:

   * Enable the XTC tools: the installation can be tested by running the command
     ``xrun --version`` from the terminal. If the command is not found, the XTC
     tools are not installed correctly.
   * Enable the Python Virtual Environment: this is checked by running the
     command ``echo %VIRTUAL_ENV%`` on Windows, or ``echo $VIRTUAL_ENV`` on
     Linux or macOS.  The path should have been set.

#. Open the notebook by running ``jupyter notebook
   lib_audio_dsp/examples/app_simple_audio_dsp_integration/dsp_design.ipynb``
   from ``lib_audio_dsp_sandbox``, as described in the 
   :ref:`Setup Steps<all_steps>` section.
