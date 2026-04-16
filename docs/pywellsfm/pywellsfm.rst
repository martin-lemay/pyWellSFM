###############################################################################
pyWellSFM Python Package
###############################################################################

pyWellSFM project contains:

* docs: documentation source
* notebooks: Jupyter notebooks that show how to use pyWellSFM features
* tests: unit and integrated tests. Tests files contain numerous examples on how to use
  pyWellSFM.
* src: pyWellSFM source code

Main API
----------

The main API can be imported directly from pywellsfm package:

.. code-block:: python

    from pywellsfm import (
        AccumulationCurve,
        AccumulationModel,
        AccumulationModelElementOptimum,
        AccumulationModelElementGaussian,
        Curve,
        DepthAgeModel,
        Element,
        EnvironmentalFacies,
        Facies,
        FaciesCriteria,
        FaciesCriteriaCollection,
        FaciesCriteriaType,
        FaciesModel,
        FSSimulator,
        Marker,
        PetrophysicalFacies,
        RealizationData,
        Scenario,
        SedimentaryFacies,
        UncertaintyCurve,
        Well,
    )

    # do anything with the imported classes and functions


Core model
^^^^^^^^^^^^

* ``pywellsfm.AccumulationModel``
* ``pywellsfm.AccumulationModelElementOptimum``
* ``pywellsfm.AccumulationModelElementGaussian``
* ``pywellsfm.AccumulationModelElementCombination``
* ``pywellsfm.Curve``
* ``pywellsfm.AccumulationCurve``
* ``pywellsfm.UncertaintyCurve``
* ``pywellsfm.DepositionalEnvironment``
* ``pywellsfm.DepositionalEnvironmentModel``
* ``pywellsfm.Element``
* ``pywellsfm.model.EnvironmentConditionModelStats``
* ``pywellsfm.model.EnvironmentConditionModelUniform``
* ``pywellsfm.model.EnvironmentConditionModelTriangular``
* ``pywellsfm.model.EnvironmentConditionModelGaussian``
* ``pywellsfm.model.EnvironmentConditionModelConstant``
* ``pywellsfm.model.EnvironmentConditionModelCurve``
* ``pywellsfm.model.EnvironmentConditionModelCombination``
* ``pywellsfm.model.EnvironmentConditionsModel``
* ``pywellsfm.Facies``
* ``pywellsfm.FaciesCriteria``
* ``pywellsfm.FaciesCriteriaCollection``
* ``pywellsfm.FaciesCriteriaType``
* ``pywellsfm.FaciesModel``
* ``pywellsfm.Marker``
* ``pywellsfm.Scenario``
* ``pywellsfm.RealizationData``
* ``pywellsfm.Well``


Method & Types
^^^^^^^^^^^^^^^^^^

* Marker type: ``pywellsfm.StratigraphicSurfaceType``

IO
^^^^^^

Loader:

* Accumulation model: :meth:`~pywellsfm.io.accumulation_model_io.loadAccumulationModel`
* Curve: :meth:`~pywellsfm.io.curve_io.loadCurvesFromFile`
* Depositional Environment Model: :meth:`~pywellsfm.io.depositional_environment_model_io.loadDepositionalEnvironmentModel`
* Environment Conditions Model: :meth:`~pywellsfm.io.environment_condition_model_io.loadEnvironmentConditionsModel`
* Facies Model: :meth:`~pywellsfm.io.facies_model_io.loadFaciesModel`
* Well: :meth:`~pywellsfm.io.well_io.loadWell`
* Simulation: :meth:`~pywellsfm.io.fssimulation_io.loadFSSimulation`
* Scenario: :meth:`~pywellsfm.io.fssimulation_io.loadScenario`
* Realization Data: :meth:`~pywellsfm.io.fssimulation_io.loadRealizationData`

Packages
-----------

.. toctree::
   :maxdepth: 1

   pywellsfm.io
   pywellsfm.model
   pywellsfm.simulator
   pywellsfm.utils