#############################
Contributing to lib_audio_dsp
#############################

This guide will focus on adding a DSP module into the library.

Module category
***************

This library tends to categorise modules into groups.
Have a quick look at the headers in ``lib_audio_dsp/api/dsp/`` to see if your module falls into any of those.
The headers roughly translate to the categories as they tend to group several APIs within them.
That header name would be roughly consistent in the documentation and the tests.
If you feel like your module doesn't fall into anything that's been done in this repo,
you can add a new header file and a documentation page for it.

C/asm API
*********

The library backend is only supported in two languages: C and assembly.
The supported file formats for those are: ``.c``, ``.s``, ``.S`` and ``.h`` for the header files. 
If you've added an API or a typedef that's intended to be public,
it should be declared in the one of the ``lib_audio_dsp/api/dsp/`` header files and have Doxygen-style comments like:

.. code-block:: C

  /**
   * @brief Struct brief
   */
  typedef struct{
    /** Param1 description */
    int32_t param1;
    /** Param2 description */
    int32_t param2;
  }your_type_t;

  /**
   * @brief API brief
   * Any additional description
   *
   * @param module    Input1 parameter description
   * @param in        Input2 parameter description
   * @return int32_t  Output type and description
   * @note Any notes if necessary
   */
  int32_t adsp_your_api(your_type_t * module, int32_t in);

Python API
**********

Python reference API is encouraged but not necessary.
Due to the nature of modern DSP algorithm development,
we tend to prototype the new modules in python before translating and optimising them.
Python reference is also often used as an extra layer of documentation providing an easy-to-look-at view of the
algorithm without going into low-level fixed point C/assembly code.
Another use of Python reference is unit testing the backend implementation.
This allows us to see the accuracy difference between double floting point and 32-bit fixed point implementations.

If you decide to implement python reference API it should live in the appropriate file in ``python/audio_dsp/dsp``.
Your python module is expected:

- to be a class which is based on the ``dsp_block`` class
- to implement at least ``__init__``, ``process`` and ``reset_state`` methods
- to have numpydoc-style docstrings for the class and the every method of it

.. code-block:: python

  import audio_dsp.dsp.generic as dspg

  class your_module(dspg.dsp_block):
  """
  Module description

  Parameters
  ----------
  param1 : float
    Input parameter description
  param2 : float
    Input parameter description

  Attributes
  ----------
  param1 : float
  param3 : float
    Attribute description

  """
  def __init__(
    self, fs: float, n_chans: int, param1: float, param2: float, Q_sig: int = dspg.Q_SIG
  ) -> None:
    super().__init__(fs, n_chans, Q_sig)
    self.param1 = param1
    self.param3 = param2 + param1

  def reset_state(self): -> None:
    """Reset module"""
    self.param1 = 0
    self.param3 = 0

  def process(self, sample: float, channel = 0) -> float:
    """
    Process description

    Parameters
    ----------
    sample : float
      The input sample to be processed.
    channel : int, optional
      The channel index to process the sample on. Default is 0.

    Returns
    -------
    float
      The processed sample.
    """
    return sample

Optionally, you can also implement ``process_xcore`` method.
``process_xcore`` tries to provide the closest implementation to the C/assembly.
Being implemented as a 32-bit fixed point version of ``process``,
``process_xcore`` is easily testable againts the backend implementation
and should have little to no accuracy difference.
``process_xcore`` can then be used to run the module without the need of the hardware.

This library uses ``ruff`` and ``pyright`` as python formatting tools.
Both of them come as pip-installable packages and are defined in the ``requirements.txt`` file.
To make sure your python code formatting passes our CI, do:

.. code-block:: console

  cd python
  make check
  make update

Documentation
*************

For the module documentation, choose an appropriate file in ``doc/05_api_reference/modules/``,
create a new heading/subheading with a link above it.
Put your documentation underneath the heading.

If you have a Python API as well as the C API you will have to use tabs with rubrics to refer those.
For the example, go to any ``.rst`` in ``doc/05_api_reference/modules/``.
Use ``doxygenstruct`` and ``doxygenfunction`` for the C API and structs and
``autoclass`` for python in the same way as in the rest of the documentation.

After your module is documented and the API is referenced it's time to add it to the components list!
To do that you need to go to ``doc/03_dsp_components/modules.rst``
and add a reference with the link to your heading.

Testing
*******

The backend C/assembly implementation has to be unit tested.
We accept two ways of doing that:

#. Testing against Python ``process`` or ``process_xcore``
#. Testing against the reference C implementation

In the second case the reference - an easy-to-look-at C API has to be implemented in the test source code.

For both cases, we expect to run (``xsim``) representative signals through the implementation and the chosen reference.
Your test should consider some egde cases as well as common representative use cases of the module.

Running tests should be done via running ``pytest -n auto``, so basic ``pytest`` structure should be built up first
(see how to wrap ``xsim`` into ``pytest`` in our current tests).
The tests have to be parallelisable, so if you intent to read and write files during your test,
you should consider using unique names for the test folders and/or file locks
(a lot of our tests already do that, so don't hesitate to take them as the example;).

Continuous Integration
**********************

Every module and test has to be added to our CI and pass before we can approve your pull request.
You are not expected to know the details or syntax of our CI system (Jenkins/Groovy).
After you raise your pull request, we will provide you with guidance on how to add your tests to Jenkins
and help with fixing it if it's a test/infrastructiure related issue.

New DSP Stages
**************

All the steps for adding a new DSP stage are listed below:

1. Add a new low level python implementation in `python/audio_dsp/dsp/`.
   - This should inherit from `dsp_block` and implement the `process` method.
   - Use fixed-point arithmetic for processing in the process_xcore method.
   - Add a basic test in `test` folder.
   - Use a similar existing implementation as a reference.
2. Make a low level C implementation in `lib_audio_dsp/api/dsp/` and `lib_audio_dsp/src/dsp`.
   - This should implement the DSP logic in C.
   - Use fixed-point arithmetic for processing.
   - Add a test in `test` folder that runs the C implementation against the
     Python implementations.
   - Use a similar existing implementation as a reference.
3. Add a new parameters class in `python/audio_dsp/stages/parameters/`.
   - This should inherit from `StageParameters`.
   - Use a similar existing implementation as a reference.
4. Create a new stage class in `python/audio_dsp/stages/`.
   - This should inherit from `Stage`.
   - Add the stage to `python/audio_dsp/stages/__init__.py`.
   - Use a similar existing implementation as a reference.
5. (Optional) Add a new placement class in `python/audio_dsp/models/placement/`.
   - This should inherit from `Placement`.
   - Use a similar existing implementation as a reference.
6. Add a stage model in `python/audio_dsp/models/stages/`.
   - This should inherit from `StageModel`.
   - This sets the parameters and placement for the stage.
   - Use a similar existing implementation as a reference.
7. Add the low level stage parameters to the yaml configuration in `stage_config`.
   - This should include the low level parameters for the stage.
   - Ensure the parameters are documented in the yaml file, including high level to low level conversions.
   - Use a similar existing implementation as a reference.
8. Add the C stage implementation in `lib_audio_dsp/api/stages/` and `lib_audio_dsp/src/stages/`.
   - This wraps the low-level C implementation in the stage wrapper.
   - This sets the memory requirements and parameters for the stage.
   - The following should be defined, with the same function API as the existing DSP stages:
      - `{stage}_state_t`: The state structure for the stage.
      - `{stage}_init`: Function to initialize the stage.
      - `{stage}_process`: Function to process the stage.
      - `{stage}_control`: Function to control the stage by updaing the {stage}_config_t structure in the {stage}_state_t.
   - Use a similar existing implementation as a reference.
9. Add a new test in `test/pipeline/test_stages.py` or `test_signal_chain_stages.py`
   that tests the new stage.
   - Use a similar existing implementation as a reference.
10. Check the module is documented in `doc\rst\05_api_reference\modules\`
    - If not, add a new documentation file in `doc/rst/05_api_reference/modules/`.
    - Ensure the documentation includes the C API, Python API, and any relevant parameters.
    - Use a similar existing implementation as a reference.
11. Lint the Python code using `ruff` and `pyright`.
    - Ensure the code is formatted correctly and passes all checks.
    - `python/Makefile` contains the linting instructions
    - the test folder does not need to be linted
