#############################
Contributing to lib_audio_dsp
#############################

This guide will focus on adding a DSP mudule into the library.

Module category
***************

This library tends to categorise modules in groups.
These categories a roughly preserved in the header file names and documentation.
Have a quick look at the headers in ``lib_audio_dsp/api/dsp/`` to see if your module falls into any of those.
That header name would be roughly consistent in the docuemntation and the tests as well.
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
Python is often used as an extra piece of documentation before going into low-level fixed point C/assembly code.
Python is also often used to unit test the C/assembly
and see the accuracy difference between double floting point against 32-bit fixed point implementations.

If you decide to implement python reference API it should live in the appropriate file in ``python/audio_dsp/dsp``.
Your python module is expected to be a class with is based on the ``dsp_block`` class
and have at least ``__init__``, ``process`` and ``reset_state`` methods.

.. code-block:: python

  import audio_dsp.dsp.generic as dspg

  class your_module(dspg.dsp_block):
  """
  Module description

  Parameters
  ----------
  param1 : float
    Input parameter description
  param1 : float
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
    return sample[channel]

Optionally, you can also implement ``process_xscope`` method.
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

Alternatively, use ``ruff`` and ``pyright`` from the command line:

.. code-block:: console

  cd python
  pyright audio_dsp --skipunannotated --level warning
  ruff check --fix
  fuff format

Documentation
*************

For the module documentation, choose an appropriate file in ``doc/05_api_reference/modules/``,
create a new heading/subheading with a link above it.
Put your documentation underneath the heading.

If you have a Python API as well as the C API you will have to use tabs with rubrics to refer those.
For the example, go to any ``.rst`` in ``doc/05_api_reference/modules/``.
Use ``doxygenstruct`` and ``doxygenfunction`` for the C API and structs and
``autoclass`` for python in the same way as in the rest of the documentation.

After your module is documented and the API is reference it's time to add it to the components list!
To do that you need to go to ``doc/03_dsp_components/modules.rst``
and add a reference with the link you just created to the appropriate place.

Testing
*******

The backend C/assembly implementation has to be unit tested.
We accept two ways of doing that:
#. Testing against Python ``process`` or ``process_xcore``
#. Testing against the reference C implementation

In the second case the reference, easy-to-look-at C API has to be implemented in the test source code.

For both cases, we expect to run (``xsim``) representative signals through the implementation and the chosen reference.
Your test should consider some egde cases and as well as common representative use cases of the module.

Running tests should be done via running ``pytest -n auto``, so basic ``pytest`` structure should be built up first
(see how to wrap ``xsim`` into pytest in our current tests).
The tests have to be parallelisable, so if you intent to read and write files during your test,
you should consider using unique names for the test folders and file locks
(a lot of our tests already do that, so don't hesitate to take them as the example).
