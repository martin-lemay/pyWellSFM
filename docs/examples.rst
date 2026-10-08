Examples
==========

The following notebooks are rendered with their stored outputs. The source
notebooks are in the `notebooks folder <https://github.com/martin-lemay/pyWellSFM/tree/main/notebooks>`_
of the repository.

Well Accommodation Calculation
*******************************

This notebook illustrates how to compute the accommodation along a well using
pyWellSFM, and compares the best estimates of accommodation (midpoint, most
likely facies water depth and smooth curve) on a synthetic well whose true
accommodation is known.

.. toctree::
   :maxdepth: 1

   notebooks/computeWellAccommodation

Stratigraphic Forward Modeling
*******************************

This notebook illustrates how to simulate sedimentary layers over time using a
stratigraphic forward modeling approach with pyWellSFM.

.. toctree::
   :maxdepth: 1

   notebooks/wellSFM

Synthetic Case Study
*********************

This notebook reproduces the synthetic case study of the pyWellSFM paper on a
rimmed carbonate platform where the truth is known. In five steps of increasing
complexity, it covers a minimal run, verification against an analytical
solution, stochastic behaviour, a multi-well transect, and the recovery of
accommodation from a degraded well with a parameter sweep.

.. toctree::
   :maxdepth: 1

   notebooks/synthetic_case_study
