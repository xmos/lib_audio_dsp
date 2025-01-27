Setup
#####

This section describes the requirements and the steps to run a basic pipeline.
This document lists the necessary steps for both Windows and Linux/macOS.
This section uses the *app_simple_audio_dsp_integration* example found within this repository.
The steps will be broadly similar for any user-created project.

.. note::

   Copying multiple lines into the console may not work as expected on Windows. 
   To avoid issues, copy and execute each line individually.

Hardware Requirements
=====================

- xcore.ai evaluation board (`XK-EVK-XU316`_ or `XK-316-AUDIO-MC-AB`_)
- xTag debugger and cable
- 2x Micro USB cable (one for power supply and one for the xTag)


Software Requirements
=====================

- `Graphviz <https://graphviz.org/download/#windows>`_: this software must
  installed and the ``dot`` executable must be on the system path.
- `XTC 15.3.0 <https://www.xmos.com/software-tools/>`_
- `Python 3.10 <https://www.python.org/downloads/>`_
- `CMake 3.21 <https://cmake.org/download/>`_

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

         call "C:\Program Files\XMOS\XTC\15.3.0\SetEnv.bat"

   .. tab:: Linux and macOS

      On Linux and macOS:

      #. Open a terminal
      #. Activate the XTC environment using *SetEnv*

      .. code-block:: console

         source /path/to/xtc/tools/SetEnv

#. Create a sandbox folder with the command below:

   .. code-block:: console

      mkdir lib_audio_dsp_sandbox

#. Clone the library inside *lib_audio_dsp_sandbox* using SSH (if you
   have shared your keys with Github) or HTTPS:

   .. code-block:: console

      cd lib_audio_dsp_sandbox

      # with SSH
      git clone git@github.com:xmos/lib_audio_dsp.git

      # without SSH
      git clone https://github.com/xmos/lib_audio_dsp.git

   For troubleshooting SSH issues, please see this
   `Github guide <https://docs.github.com/en/authentication/troubleshooting-ssh>`_.

#. Get the sandbox inside *lib_audio_dsp_sandbox*. This step can take several
   minutes.

   .. tab:: Windows

      On Windows:

      .. code-block:: console

         cd lib_audio_dsp/examples/app_simple_audio_dsp_integration
         cmake -B build -G Ninja 
         cd ../../..

   .. tab:: Linux and macOS

      On Linux and macOS:

      .. code-block:: console

         cd lib_audio_dsp/examples/app_simple_audio_dsp_integration 
         cmake -B build 
         cd ../../..

#. Create a Python virtualenv inside *lib_audio_dsp_sandbox*, and install
   lib_audio_dsp and it's requirements.

   .. tab:: Windows

      On Windows:

      .. code-block:: console

         py -3.10 -m venv .venv 
         call .venv/Scripts/activate.bat 
         pip install -e ./lib_audio_dsp/python

   .. tab:: Linux and macOS

      On Linux and macOS:

      .. code-block:: console

         python3.10 -m venv .venv 
         source .venv/bin/activate 
         pip install -e ./lib_audio_dsp/python

#. Connect an XCORE-AI-EXPLORER using both USB ports

#. The examples are presented as a Jupyter notebook for interactive development.
   Install Juptyer notebooks into the Python virtual environment with the command:

   .. code-block:: console

      pip install notebook==7.2.1

#. Open the notebook by running from *lib_audio_dsp_sandbox* the following
   command:

   .. code-block:: console

      jupyter notebook lib_audio_dsp/examples/app_simple_audio_dsp_integration/dsp_design.ipynb

   If a blank screen appears or nothing opens, then copy the link starting with
   http://127.0.0.1/ from the terminal into the browser. The top level Jupyter
   notebook page should open, as can be seein in :numref:`top_level_notebook`.

   .. _top_level_notebook:

   .. figure:: ../images/jupyter_notebook_top_level.png
      :width: 25%

      Top-level page of the Jupyter Notebook

#. Run all the cells from the browser. From the menu at the top of the page
   click *Run -> Run all cells* (:numref:`run_all_cells`).
   This creates the pipeline and builds the app. Wait for all the cells to
   finish.

   .. _run_all_cells:

   .. figure:: ../images/jupyter_notebook_run_tests.png
      :width: 100%

      Run menu of the Jupyter Notebook

   Any configuration or compilation errors will be displayed in the notebook in
   the *Build and run* cell, as in the example on :numref:`run_error`.

   .. _run_error:

   .. figure:: ../images/config_error.png
      :width: 100%

      Run error of the Jupyter Notebook

#. Update and run *Pipeline design stage* to add the desired audio processing
   blocks. A diagram will be generated showing the pipeline IO mapping.
   A simple pipeline example is shown in :numref:`pipeline_diagram`.
   See the top of the notebook for more information about this stage.

   .. _pipeline_diagram:

   .. figure:: ../images/pipeline_diagram.png
      :width: 25%

      Diagram of a simple audio pipeline

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
   * From your sandbox, enable the Python Virtual Environment and check the path is set:
   
   .. tab:: Windows

      On Windows:

      .. code-block:: console

         call .venv/Scripts/activate.bat 
         echo %VIRTUAL_ENV%

   .. tab:: Linux and macOS

      On Linux and macOS:

      .. code-block:: console

         source .venv/bin/activate
         echo $VIRTUAL_ENV

#. Open the notebook by running ``jupyter notebook
   lib_audio_dsp/examples/app_simple_audio_dsp_integration/dsp_design.ipynb``
   from ``lib_audio_dsp_sandbox``, as described in the 
   :ref:`Setup Steps<all_steps>` section.
