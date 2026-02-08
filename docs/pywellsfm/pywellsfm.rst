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
        AccumulationModelBase,
        AccumulationModelEnvironmentOptimum,
        AccumulationModelGaussian,
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
        FSSimulatorRunner,
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

* :class:`~pywellsfm.model.AccumulationModel.AccumulationModelBase`
* :class:`~pywellsfm.model.AccumulationModel.AccumulationModelEnvironmentOptimum`
* :class:`~pywellsfm.model.AccumulationModel.AccumulationModelGaussian`
* :class:`~pywellsfm.model.Curve.Curve`
* :class:`~pywellsfm.model.Curve.AccumulationCurve`
* :class:`~pywellsfm.model.Curve.UncertaintyCurve`
* :class:`~pywellsfm.model.Element.Element`
* :class:`~pywellsfm.model.Facies.Facies`
* :class:`~pywellsfm.model.Facies.FaciesCriteria`
* :class:`~pywellsfm.model.Facies.FaciesCriteriaCollection`
* :class:`~pywellsfm.model.Facies.FaciesCriteriaType`
* :class:`~pywellsfm.model.Facies.FaciesModel`
* :class:`~pywellsfm.model.Marker.Marker`
* :class:`~pywellsfm.model.SimulationParameters.Scenario`
* :class:`~pywellsfm.model.SimulationParameters.SimulationData`
* :class:`~pywellsfm.model.SimulationParameters.RealizationData`
* :class:`~pywellsfm.model.Well.Well`


Method & Types
^^^^^^^^^^^^^^^^^^

* Marker type: :class:`~pywellsfm.model.Marker.StratigraphicSurfaceType`


IO
^^^^^^

Loader:

* Accumulation model: :meth:`~pywellsfm.io.accumulation_model.loadAccumulationModel`
* Curve: :meth:`~pywellsfm.io.curve.loadCurvesFromFile`
* Facies Model: :meth:`~pywellsfm.io.loadFaciesModel.`
* Well: :meth:`~pywellsfm.io.well.loadWell`
* Simulation: :meth:`~pywellsfm.io.simulation.loadSimulationData`
* Scenario: :meth:`~pywellsfm.io.simulation.loadScenario`
* Realization Data: :meth:`~pywellsfm.io.simulation.loadRealizationData`

Packages
-----------

.. toctree::
   :maxdepth: 1

   pywellsfm.io
   pywellsfm.model
   pywellsfm.simulator
   pywellsfm.utils