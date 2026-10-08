# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file
import os
import sys

m_path = os.path.dirname(os.getcwd())
if m_path not in sys.path:
    sys.path.insert(0, os.path.join(m_path, "src"))

import numpy as np
import numpy.typing as npt

from pywellsfm.model import (
    AccumulationCurve,
    AccumulationModel,
    AccumulationModelElementOptimum,
    CarbonateOpenRampDepositionalEnvironmentModel,
    Curve,
    Marker,
    RealizationData,
    Scenario,
    SubsidenceType,
    Well,
)
from pywellsfm.simulator import FSSimulator


def _environments(seed: int | None) -> npt.NDArray[np.str_]:
    """Run a small ramp simulation and return simulated environments."""
    curve = AccumulationCurve(
        "waterDepth",
        np.array([-1e4, 0.0, 10.0, 30.0, 1e4]),
        np.array([0.0, 1.0, 1.0, 0.0, 0.0]),
    )
    model = AccumulationModel(
        "m", {"A": AccumulationModelElementOptimum("A", 50.0, {"wd": curve})}
    )
    scenario = Scenario(
        "s",
        model,
        None,
        CarbonateOpenRampDepositionalEnvironmentModel(),
    )
    well = Well("W", np.zeros(3), 100.0)
    well.addMarkers([Marker("Base", 100.0, 1.0), Marker("Top", 0.0, 0.0)])
    subsidence = Curve(
        "Age", "Subsidence", np.array([0.0, 2.0]), np.array([40.0, 40.0])
    )
    realization = RealizationData(
        well, 10.0, None, subsidence, SubsidenceType.RATE
    )
    simulator = FSSimulator(
        scenario,
        [realization, realization],
        use_depositional_environment_simulator=True,
        seed=seed,
    )
    simulator.prepare()
    simulator.run()
    simulator.finalize()
    assert simulator.outputs is not None
    return np.asarray(simulator.outputs["environment"].values)


def test_same_seed_gives_same_environments() -> None:
    """Two runs with the same seed give identical environment sequences."""
    np.testing.assert_array_equal(_environments(3), _environments(3))


def test_different_seeds_give_different_environments() -> None:
    """Different seeds give different environment sequences."""
    assert not np.array_equal(_environments(3), _environments(4))
