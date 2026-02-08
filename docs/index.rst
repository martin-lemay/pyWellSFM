Python Well SFM - pyWellSFM
=============================

``pyWellSFM``, which stands for Python Well SFM (Stratigraphic Forward Modeling), is a
``Python`` package dedicated to the analysis of accommodation along wells and the
simulation of sedimentary layers over time. 

``pyWellSFM`` consists in:

- a data structure that stores well information (e.g. depth, age, lithology, etc.) and
  the results of the accommodation analysis and simulations
- tools to compute accommodation along wells
- tools to simulate sedimentary layers over time using a stratigraphic forward modeling
  approach
- tools to visualize the results of the accommodation analysis and simulations


.. NOTE::

   If you use ``pyWellSFM``, please refer to the GitHub repository and cite the software as follows:

   - pyWellSFM: Lemay, M. (2026). pyWellSFM: Python Well Stratigraphic Forward
     Modeling (v0.0.1). https://github.com/martin-lemay/pyWellSFM


Installation
-------------

To install ``pyWellSFM``, you can clone the ``GitHub`` `repository <https://github.com/martin-lemay/pyWellSFM.git>`_.
It is recommended to use a 
`virtual Python environment <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments>`_.

Run the following commands:
- using pip and a virtual environment:

.. code-block:: bash
   
   cd path/to/install/dir/
   git clone https://github.com/martin-lemay/pyWellSFM.git
   cd pyWellSFM
   source .venv/bin/activate
   pip install -e ./

- using conda:

.. code-block:: bash
   
   cd path/to/install/dir/
   git clone https://github.com/martin-lemay/pyWellSFM.git
   cd pyWellSFM
   conda activate venv
   conda install ./

Testing
---------

You can test pyWellSFM package using ``pytest`` (see the `homepage <https://pytest.org/>`_).

* To test the source distribution, run the following commands from pyWellSFM root directory:

.. code-block:: bash

   pytest ./

* To test the installed package, run the following commands:

.. code-block:: bash

   pytest --pyargs pywellsfm


Contributing
--------------

Contributions are welcome — bug reports, feature requests, docs improvements, and code
changes.

Workflow (issues + PR/MR)
^^^^^^^^^^^^^^^^^^^^^^^^^

- Create an **issue** first to describe the bug / enhancement (with a minimal
   reproducible example when relevant).
- Create a **Pull Request / Merge Request** that **addresses one issue**.

  - Reference the issue in the PR description (e.g. ``Fixes #123``).
  - Keep changes focused and include tests/docs updates when applicable.

Local setup
^^^^^^^^^^^

.. code-block:: bash

   pip install -e .[dev,test]

If you plan to build the docs locally, install the doc build dependencies as well:

.. code-block:: bash

   pip install -r requirements.txt

Formatting, linting, typing, tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run these from the repository root:

.. code-block:: bash

   # Format
   ruff format .

   # Lint (optionally auto-fix)
   ruff check .
   ruff check --fix .

   # Type-check
   mypy .

   # Tests
   pytest

   # To mirror CI more closely (includes doctests)
   pytest ./ --doctest-modules

Build the docs locally
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m sphinx -b html docs docs/_build/html

Then open ``docs/_build/html/index.html`` in your browser.

What is checked in CI
^^^^^^^^^^^^^^^^^^^^^

On each Pull Request, GitHub Actions runs:

- ``ruff check`` (lint; currently non-blocking in CI)
- ``mypy`` (static type checks)
- ``pytest`` (tests + doctests)
- ``sphinx`` (documentation build)

Contents
----------

.. toctree::
   :maxdepth: 1

   user_guide/user_guide
   examples
   pywellsfm/pywellsfm   

