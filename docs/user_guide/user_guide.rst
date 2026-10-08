User Guide
==========

.. WARNING::

    This section is a work in progress.

    See also:

    * the notebooks in the Examples section
    * the API reference under Packages

Accommodation along wells
-------------------------

:class:`~pywellsfm.model.AccommodationSpaceWellCalculator.AccommodationSpaceWellCalculator`
computes the apparent accommodation from a facies log: the deposited thickness
since the base plus the change of water depth, the water depth being known from
the ``waterDepth`` criteria of each facies.

.. code-block:: python

    from pywellsfm import (
        AccommodationEstimateMethod,
        AccommodationSpaceWellCalculator,
    )

    calculator = AccommodationSpaceWellCalculator(well, faciesList)
    curve = calculator.computeAccommodationCurve(
        "lithology", estimateMethod=AccommodationEstimateMethod.SMOOTH
    )

The output is an uncertainty curve:

* the **minimum and maximum curves** bracket the accommodation. Their width is
  set by the water depth ranges of the facies, by the water depth at the base
  (``waterDepthAtBase``) and by the rule used at facies boundaries
  (``boundaryRule``, union or intersection of the ranges of both facies);
* the **median curve** is the best estimate inside this range, chosen with
  ``estimateMethod``:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Method
     - Best estimate
   * - ``MIDPOINT``
     - Middle of the accommodation range at each facies boundary (default).
   * - ``MODE``
     - Most likely water depth of each facies (mode of the ``waterDepth``
       criteria). At a boundary, the modes of both facies are averaged and
       limited to the water depth range of the boundary.
   * - ``SMOOTH``
     - Smoothest accommodation curve inside the range, close to the facies
       modes. ``smoothingLength`` is the wavelength (same unit as depth) below
       which accommodation variations are damped.

Most likely facies water depth
******************************

The water depth of a facies is a range, but some depths are more likely than
others (e.g., from modern analogues). The most likely value is the ``mode`` of
the criteria, used by the ``MODE`` and ``SMOOTH`` methods. Without mode, the
distribution is assumed uniform and the middle of the range is used.

.. code-block:: python

    from pywellsfm import FaciesCriteria, FaciesCriteriaType, SedimentaryFacies

    sandstone = SedimentaryFacies(
        "sandstone",
        {
            FaciesCriteria(
                "waterDepth",
                0.0,
                10.0,
                FaciesCriteriaType.SEDIMENTOLOGICAL,
                mode=6.0,
            )
        },
    )

In a facies model JSON file, the mode is the optional ``mode`` key of a
criteria:

.. code-block:: json

    {
        "name": "waterDepth",
        "minRange": 0.0,
        "maxRange": 10.0,
        "mode": 6.0
    }
