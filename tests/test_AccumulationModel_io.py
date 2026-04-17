# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

"""Test suite for AccumulationModel classes and unified API."""

import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

# Add src to path for imports
m_path = os.path.dirname(os.getcwd())
if m_path not in sys.path:
    sys.path.insert(0, os.path.join(m_path, "src"))

import pywellsfm.io.accumulation_model_io as accumulation_model_io
from pywellsfm.io import (
    loadAccumulationModel,
    loadAccumulationModelGaussianFromCsv,
    saveAccumulationModelEnvironmentOptimumToJson,
    saveAccumulationModelGaussianToCsv,
    saveAccumulationModelGaussianToJson,
)
from pywellsfm.io.accumulation_model_io import (
    _loadAccumulationCurveFromCurveJsonObj,
    accumulationModelEnvironmentOptimumToJsonObjInline,
    accumulationModelGaussianToJsonObj,
    accumulationModelToJsonObj,
    loadAccumulationModelFromJsonObj,
    saveAccumulationModel,
)
from pywellsfm.io.curve_io import curveToJsonObj
from pywellsfm.model import AccumulationCurve
from pywellsfm.model.AccumulationModel import (
    AccumulationModel,
    AccumulationModelCombination,
    AccumulationModelElementGaussian,
    AccumulationModelElementOptimum,
)

# ------------------------------
# Helpers for ioHelpers tests
# ------------------------------


def _write_json(tmp_path: Path, payload: dict[str, Any], filename: str) -> str:
    """Helper: write JSON to a temp file and return its filesystem path."""
    path = tmp_path / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _write_csv(
    tmp_path: Path, rows: list[list[float | str]], filename: str
) -> str:
    """Helper: write a simple CSV file and return its filesystem path."""
    path = tmp_path / filename
    lines = [",".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _gaussian_signature(
    model: AccumulationModel,
) -> dict[str, tuple[float, float]]:
    """Canonical signature used by tests to compare Gaussian models."""
    out: dict[str, tuple[float, float]] = {}
    for element_name, element_model in model.elements.items():
        assert isinstance(element_model, AccumulationModelElementGaussian)
        out[element_name] = (
            float(element_model.accumulationRate),
            float(element_model.std_dev_factor),
        )
    return dict(sorted(out.items()))


def _envopt_signature(
    model: AccumulationModel,
) -> tuple[dict[str, float], dict[str, tuple[list[float], list[float]]]]:
    """Canonical signature to compare EnvironmentOptimum models."""
    elements: dict[str, float] = {}
    curves: dict[str, tuple[list[float], list[float]]] = {}
    for element_name, element_model in model.elements.items():
        assert isinstance(element_model, AccumulationModelElementOptimum)
        elements[element_name] = float(element_model.accumulationRate)
        for curve_name, curve in element_model.accumulationCurves.items():
            curves[curve_name] = (
                [float(v) for v in curve._abscissa],
                [float(v) for v in curve._ordinate],
            )
    return dict(sorted(elements.items())), dict(sorted(curves.items()))


# -------------------------------------------------
# ioHelpers: Gaussian model (CSV)
# -------------------------------------------------


def test_loadAccumulationModelGaussianFromCsv_happy_path(
    tmp_path: Path,
) -> None:
    """Test loading a Gaussian accumulation model from a CSV file.

    Objective:
    - Ensure `loadAccumulationModelGaussianFromCsv` parses the expected columns
      and creates the corresponding `AccumulationModelGaussian` with elements
      and per-element stddev factors.

    Input data:
    - A temporary CSV file with header: name, mean, stddevFactor.
    - Two elements: sand (100.0, 0.2) and shale (50.0, 0.1).

    Expected outputs:
    - Returned model is an `AccumulationModelGaussian`.
    - Elements contain both names with matching accumulation rates.
    - std_dev_factors contain both names with matching factors.
    """
    csv_path = _write_csv(
        tmp_path,
        [
            ["name", "accumulationRate", "stddevFactor"],
            ["sand", 100.0, 0.2],
            ["shale", 50.0, 0.1],
        ],
        "gaussian.csv",
    )

    model = loadAccumulationModelGaussianFromCsv(csv_path)

    assert isinstance(model, AccumulationModel)
    sig = _gaussian_signature(model)
    assert sig == {"sand": (100.0, 0.2), "shale": (50.0, 0.1)}


def test_saveAccumulationModelGaussianToCsv_round_trip(tmp_path: Path) -> None:
    """Test exporting then reloading a Gaussian model via CSV preserves values.

    Objective:
    - Verify `saveAccumulationModelGaussianToCsv` produces a CSV that
      round-trips through `loadAccumulationModelGaussianFromCsv` without losing
      element means or stddev factors.

    Input data:
    - A Gaussian model with two elements and explicit stddev factors.

    Expected outputs:
    - After export + load, the signature (element rate + stddev factor) is
      identical.
    """
    model = AccumulationModel(
        "MyGaussian",
        {
            "sand": AccumulationModelElementGaussian("sand", 100.0, 0.25),
            "shale": AccumulationModelElementGaussian("shale", 50.0, 0.1),
        },
    )

    out_csv = tmp_path / "gaussian_out.csv"
    saveAccumulationModelGaussianToCsv(model, str(out_csv))

    reloaded = loadAccumulationModelGaussianFromCsv(str(out_csv))
    assert _gaussian_signature(reloaded) == _gaussian_signature(model)


# -------------------------------------------------
# ioHelpers: Gaussian model (JSON)
# -------------------------------------------------


def test_loadAccumulationModelGaussian_happy_path(tmp_path: Path) -> None:
    """Test loading a Gaussian accumulation model from schema-shaped JSON.

    Objective:
    - Ensure `loadAccumulationModel` accepts a schema-compliant JSON
      payload and builds the correct model.

    Input data:
        - A JSON file matching `AccumulationModelSchema.json` with per-element
            modelType="Gaussian".

    Expected outputs:
    - Returned model has the expected name, elements, and stddev factors.
    """
    payload: dict[str, Any] = {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": "G",
            "elements": {
                "sand": {
                    "accumulationRate": 100.0,
                    "model": {
                        "modelType": "Gaussian",
                        "stddevFactor": 0.2,
                    },
                },
                "shale": {
                    "accumulationRate": 50.0,
                    "model": {
                        "modelType": "Gaussian",
                        "stddevFactor": 0.1,
                    },
                },
            },
        },
    }

    json_path = _write_json(tmp_path, payload, "gaussian.json")
    model = cast(AccumulationModel, loadAccumulationModel(json_path))

    assert model.name == "G"
    assert _gaussian_signature(model) == {
        "sand": (100.0, 0.2),
        "shale": (50.0, 0.1),
    }


def test_save_AccumulationModel_round_trip(tmp_path: Path) -> None:
    """Test exporting/reloading a Gaussian model via JSON preserves values.

    Objective:
    - Verify `saveAccumulationModelGaussianToJson` produces a JSON file that
      round-trips through `loadAccumulationMode without losing
      element rates or stddev factors.

    Input data:
    - A Gaussian model with two elements and explicit stddev factors.

    Expected outputs:
    - After export + load, the signature is identical.
    """
    model = AccumulationModel(
        "MyGaussian",
        {
            "sand": AccumulationModelElementGaussian("sand", 100.0, 0.25),
            "shale": AccumulationModelElementGaussian("shale", 50.0, 0.1),
        },
    )

    out_json = tmp_path / "gaussian_out.json"
    saveAccumulationModelGaussianToJson(model, str(out_json))

    reloaded = cast(AccumulationModel, loadAccumulationModel(str(out_json)))
    assert reloaded.name == "MyGaussian"
    assert _gaussian_signature(reloaded) == _gaussian_signature(model)


@pytest.mark.parametrize(
    "payload, expected_message",
    [
        ([], "must be an object"),
        (
            {
                "format": "x",
                "version": "1.0",
                "accumulationModel": {
                    "name": "G",
                    "elements": {},
                },
            },
            "Invalid accumulation model format",
        ),
        (
            {
                "format": "pyWellSFM.AccumulationModelData",
                "version": "x",
                "accumulationModel": {
                    "name": "G",
                    "elements": {},
                },
            },
            "Invalid accumulation model version",
        ),
        (
            {
                "format": "pyWellSFM.AccumulationModelData",
                "version": "1.0",
                "accumulationModel": "not-an-object",
            },
            "'accumulationModel' must be an object",
        ),
        (
            {
                "format": "pyWellSFM.AccumulationModelData",
                "version": "1.0",
                "accumulationModel": {
                    "name": "G",
                    "elements": {},
                },
            },
            "must be a non-empty object",
        ),
    ],
)
def test_loadAccumulationModel_rejects_invalid_payloads(
    tmp_path: Path,
    payload: Any,  # noqa: ANN401
    expected_message: str,
) -> None:
    """Test JSON loader rejects invalid shapes/metadata.

    Objective:
    - Verify `loadAccumulationModel` enforces the top-level metadata
      (format/version) and the expected object shapes and modelType.

    Input data:
    - A set of invalid payloads (wrong type, wrong format/version,
      wrong modelType).

    Expected outputs:
    - Each invalid payload raises ValueError with a message containing the
      expected substring.
    """
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        loadAccumulationModel(str(path))
    assert expected_message in str(exc.value)


# -------------------------------------------------
# ioHelpers: EnvironmentOptimum model (JSON)
# -------------------------------------------------


def test_loadAccumulationModelEnvironmentOptimumFromJson_inline_curves(
    tmp_path: Path,
) -> None:
    """Test loading an EnvironmentOptimum element model with inline curves.

    Objective:
    - Ensure `loadAccumulationModelEnvironmentOptimumFromJson` supports inline
      TabulatedFunction objects inside `accumulationCurves`.

    Input data:
    - A schema-shaped EnvironmentOptimum JSON file with one element and two
      inline curves (WaterDepth and Energy).

    Expected outputs:
    - Returned model contains the element with correct accumulationRate.
    - prodCurves contains both curves keyed by abscissaName.
    - Curve interpolation behaves as expected at an intermediate point.
    """
    bathy_curve_obj = curveToJsonObj(
        AccumulationCurve(
            "WaterDepth",
            np.array([0.0, 10.0]),
            np.array([0.0, 1.0]),
        ),
        y_axis_name="ReductionCoeff",
        x_axis_name_default="WaterDepth",
    )
    energy_curve_obj = curveToJsonObj(
        AccumulationCurve(
            "Energy",
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ),
        y_axis_name="ReductionCoeff",
        x_axis_name_default="Energy",
    )

    payload: dict[str, Any] = {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": "Env",
            "elements": {
                "sand": {
                    "accumulationRate": 10.0,
                    "model": {
                        "modelType": "EnvironmentOptimum",
                        "accumulationCurves": [
                            bathy_curve_obj,
                            energy_curve_obj,
                        ],
                    },
                }
            },
        },
    }

    json_path = _write_json(tmp_path, payload, "env_inline.json")
    model = cast(AccumulationModel, loadAccumulationModel(json_path))

    assert model.name == "Env"
    sand_model = model.getElementModel("sand")
    assert isinstance(sand_model, AccumulationModelElementOptimum)
    assert "waterdepth" in sand_model.accumulationCurves
    assert "energy" in sand_model.accumulationCurves

    # Linear curve between 0 and 10 -> at 5 = 0.5
    assert sand_model.getAccumulationAt({"WaterDepth": 5.0}) == pytest.approx(
        10.0 * 0.5
    )


def test_loadAccumulationModelEnvironmentOptimumFromJson_url_curves(
    tmp_path: Path,
) -> None:
    """Test loading an EnvironmentOptimum element model with curves by url.

    Objective:
    - Ensure `accumulationCurves` supports schema refs: {"url": "curve.json"}
      resolved relative to the accumulation model JSON file location.

    Expected outputs:
    - Returned model contains the curve loaded from file.
    """
    bathy_curve_obj = curveToJsonObj(
        AccumulationCurve(
            "WaterDepth",
            np.array([0.0, 10.0]),  # WaterDepth array
            np.array([0.0, 1.0]),  # ReductionCoeff array,
        ),
        y_axis_name="ReductionCoeff",
        x_axis_name_default="WaterDepth",
    )
    curve_path = tmp_path / "bathy_curve.json"
    curve_path.write_text(
        json.dumps(bathy_curve_obj, indent=2), encoding="utf-8"
    )

    payload: dict[str, Any] = {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": "Env",
            "elements": {
                "sand": {
                    "accumulationRate": 10.0,
                    "model": {
                        "modelType": "EnvironmentOptimum",
                        "accumulationCurves": [
                            {"url": "bathy_curve.json"},
                        ],
                    },
                }
            },
        },
    }

    json_path = _write_json(tmp_path, payload, "env_url.json")
    model = cast(AccumulationModel, loadAccumulationModel(json_path))

    sand_model = model.getElementModel("sand")
    assert isinstance(sand_model, AccumulationModelElementOptimum)
    assert "waterdepth" in sand_model.accumulationCurves
    assert sand_model.getAccumulationAt({"WaterDepth": 5.0}) == pytest.approx(
        10.0 * 0.5
    )


def test_saveAccumulationModelEnvironmentOptimumToJson_inline_round_trip(
    tmp_path: Path,
) -> None:
    """Test exporting an EnvironmentOptimum model with inline round-trips.

    Objective:
    - Verify
      `saveAccumulationModelEnvironmentOptimumToJson(curves_mode='inline')`
      produces a single JSON file that loads back to an equivalent model.

    Input data:
    - An EnvironmentOptimum model with 2 elements and 2 curves.

    Expected outputs:
    - Loading the exported JSON yields same element rates and curve tabulation.
    """
    model = AccumulationModel(
        "Env",
        {
            "sand": AccumulationModelElementOptimum(
                "sand",
                10.0,
                accumulationCurves={
                    "WaterDepth": AccumulationCurve(
                        "WaterDepth",
                        np.array([0.0, 10.0]),
                        np.array([0.0, 1.0]),
                    ),
                    "Energy": AccumulationCurve(
                        "Energy",
                        np.array([0.0, 1.0]),
                        np.array([0.0, 1.0]),
                    ),
                },
            ),
            "shale": AccumulationModelElementOptimum(
                "shale",
                2.5,
                accumulationCurves={
                    "WaterDepth": AccumulationCurve(
                        "WaterDepth",
                        np.array([0.0, 10.0]),
                        np.array([0.0, 1.0]),
                    ),
                    "Energy": AccumulationCurve(
                        "Energy",
                        np.array([0.0, 1.0]),
                        np.array([0.0, 1.0]),
                    ),
                },
            ),
        },
    )

    out_json = tmp_path / "env_inline_out.json"
    saveAccumulationModelEnvironmentOptimumToJson(
        model, str(out_json), curves_mode="inline"
    )

    reloaded = cast(
        AccumulationModel,
        loadAccumulationModel(str(out_json)),
    )
    assert _envopt_signature(reloaded) == _envopt_signature(model)


def test_saveAccumulationModelEnvironmentOptimumToJson_external_raises(
    tmp_path: Path,
) -> None:
    """External curve mode is not supported by the current schema."""
    bathy_curve = AccumulationCurve(
        "WaterDepth",
        np.array([0.0, 10.0, 50.0]),
        np.array([0.0, 1.0, 0.8]),
    )
    model = AccumulationModel(
        "Env",
        {
            "sand": AccumulationModelElementOptimum(
                "sand",
                10.0,
                accumulationCurves={"WaterDepth": bathy_curve},
            )
        },
    )
    out_json = tmp_path / "env_external_out.json"
    with pytest.raises(ValueError):
        saveAccumulationModelEnvironmentOptimumToJson(
            model,
            str(out_json),
            curves_mode="external",
        )


def test__loadAccumulationCurveFromCurveJsonObj_rejects_invalid_ordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AccumulationCurve conversion rejects ordinates outside [0, 1]."""

    class _FakeCurve:
        _xAxisName = "WaterDepth"
        _abscissa = np.array([0.0, 1.0])
        _ordinate = np.array([-0.1, 1.1])

    def _fake_load_curve_from_json_obj(obj: dict[str, Any]) -> Any:  # noqa: ANN401
        return _FakeCurve()

    monkeypatch.setattr(
        accumulation_model_io,
        "loadCurveFromJsonObj",
        _fake_load_curve_from_json_obj,
    )

    curve_obj: dict[str, Any] = {"format": "pyWellSFM.CurveData"}

    with pytest.raises(ValueError, match="must be between 0 and 1"):
        _loadAccumulationCurveFromCurveJsonObj(curve_obj)


def test_accumulationModelGaussianToJsonObj_rejects_non_gaussian() -> None:
    """Gaussian serializer refuses mixed model element types."""
    model = AccumulationModel(
        "Mixed",
        {
            "sand": AccumulationModelElementGaussian("sand", 10.0, 0.1),
            "shale": AccumulationModelElementOptimum("shale", 2.0),
        },
    )

    with pytest.raises(
        ValueError, match="requires all elements to be Gaussian"
    ):
        accumulationModelGaussianToJsonObj(model)


def test_AMEnvironmentOptimumToJsonObjInline_rejects_non_optimum() -> None:
    """EnvironmentOptimum serializer refuses non optimum element models."""
    model = AccumulationModel(
        "Mixed",
        {
            "sand": AccumulationModelElementOptimum("sand", 10.0),
            "shale": AccumulationModelElementGaussian("shale", 2.0, 0.2),
        },
    )

    with pytest.raises(
        ValueError,
        match="requires all elements to be EnvironmentOptimum",
    ):
        accumulationModelEnvironmentOptimumToJsonObjInline(model)


def test_accumulationModelToJsonObj_rejects_invalid_curves_type() -> None:
    """Serializer validates EnvironmentOptimum accumulationCurves type."""
    elem = AccumulationModelElementOptimum("sand", 10.0)
    elem.accumulationCurves = cast(Any, [])
    model = AccumulationModel("Env", {"sand": elem})

    with pytest.raises(ValueError, match="accumulationCurves must be a dict"):
        accumulationModelToJsonObj(model)


def test_accumulationModelToJsonObj_rejects_unsupported_element_type() -> None:
    """Generic serializer rejects unsupported accumulation model subclasses."""
    combo = AccumulationModelCombination(
        [AccumulationModelElementGaussian("sand", 10.0, 0.2)]
    )
    model = AccumulationModel("Combo", {"sand": combo})

    with pytest.raises(
        ValueError, match="Unsupported element accumulation model type"
    ):
        accumulationModelToJsonObj(model)


def test_loadAccumulationModel_rejects_unsupported_file_extension(
    tmp_path: Path,
) -> None:
    """Top-level loader rejects unsupported file extension."""
    txt_path = tmp_path / "acc_model.txt"
    txt_path.write_text("not used", encoding="utf-8")

    with pytest.raises(
        ValueError, match="Unsupported accumulation model file format"
    ):
        loadAccumulationModel(str(txt_path))


@pytest.mark.parametrize(
    "payload, expected_message",
    [
        (
            {
                "format": "pyWellSFM.AccumulationModelData",
                "version": "1.0",
                "accumulationModel": {
                    "name": "   ",
                    "elements": {
                        "sand": {
                            "accumulationRate": 1.0,
                            "model": {
                                "modelType": "Gaussian",
                                "stddevFactor": 0.2,
                            },
                        }
                    },
                },
            },
            "name must be a non-empty string",
        ),
        (
            {
                "format": "pyWellSFM.AccumulationModelData",
                "version": "1.0",
                "accumulationModel": {
                    "name": "G",
                    "elements": {
                        1: {
                            "accumulationRate": 1.0,
                            "model": {
                                "modelType": "Gaussian",
                                "stddevFactor": 0.2,
                            },
                        }
                    },
                },
            },
            "keys must be non-empty strings",
        ),
        (
            {
                "format": "pyWellSFM.AccumulationModelData",
                "version": "1.0",
                "accumulationModel": {
                    "name": "G",
                    "elements": {"sand": "not-an-object"},
                },
            },
            "must be an object",
        ),
        (
            {
                "format": "pyWellSFM.AccumulationModelData",
                "version": "1.0",
                "accumulationModel": {
                    "name": "G",
                    "elements": {
                        "sand": {
                            "accumulationRate": "bad",
                            "model": {
                                "modelType": "Gaussian",
                                "stddevFactor": 0.2,
                            },
                        }
                    },
                },
            },
            "accumulationRate must be a number",
        ),
        (
            {
                "format": "pyWellSFM.AccumulationModelData",
                "version": "1.0",
                "accumulationModel": {
                    "name": "G",
                    "elements": {
                        "sand": {
                            "accumulationRate": 1.0,
                            "model": "not-an-object",
                        }
                    },
                },
            },
            ".model must be an object",
        ),
        (
            {
                "format": "pyWellSFM.AccumulationModelData",
                "version": "1.0",
                "accumulationModel": {
                    "name": "G",
                    "elements": {
                        "sand": {
                            "accumulationRate": 1.0,
                            "model": {
                                "modelType": "Gaussian",
                                "stddevFactor": "bad",
                            },
                        }
                    },
                },
            },
            "stddevFactor must be a number",
        ),
        (
            {
                "format": "pyWellSFM.AccumulationModelData",
                "version": "1.0",
                "accumulationModel": {
                    "name": "E",
                    "elements": {
                        "sand": {
                            "accumulationRate": 1.0,
                            "model": {
                                "modelType": "EnvironmentOptimum",
                                "accumulationCurves": [],
                            },
                        }
                    },
                },
            },
            "accumulationCurves must be a non-empty array",
        ),
        (
            {
                "format": "pyWellSFM.AccumulationModelData",
                "version": "1.0",
                "accumulationModel": {
                    "name": "G",
                    "elements": {
                        "sand": {
                            "accumulationRate": 1.0,
                            "model": {
                                "modelType": "Other",
                            },
                        }
                    },
                },
            },
            "modelType must be 'Gaussian' or 'EnvironmentOptimum'",
        ),
    ],
)
def test_loadAccumulationModelFromJsonObj_validation_errors(
    payload: dict[str, Any],
    expected_message: str,
) -> None:
    """Direct JSON-object loader validates schema and model variants."""
    with pytest.raises(ValueError, match=expected_message):
        loadAccumulationModelFromJsonObj(payload)


def test_loadAMFromJsonObj_rejects_inline_curve_wrong_format() -> None:
    """EnvironmentOptimum inline curves must use CurveSchema format tag."""
    payload: dict[str, Any] = {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": "Env",
            "elements": {
                "sand": {
                    "accumulationRate": 1.0,
                    "model": {
                        "modelType": "EnvironmentOptimum",
                        "accumulationCurves": [
                            {
                                "format": "Wrong.Format",
                            }
                        ],
                    },
                }
            },
        },
    }

    with pytest.raises(
        ValueError, match="must be either an inline CurveSchema"
    ):
        loadAccumulationModelFromJsonObj(payload)


def test_loadAMFromJsonObj_rejects_conflicting_duplicate_curves() -> None:
    """Duplicate curve names are allowed only when tabulated values match."""
    c1 = curveToJsonObj(
        AccumulationCurve(
            "WaterDepth",
            np.array([0.0, 10.0]),
            np.array([0.0, 1.0]),
        ),
        y_axis_name="ReductionCoeff",
        x_axis_name_default="WaterDepth",
    )
    c2 = curveToJsonObj(
        AccumulationCurve(
            "WaterDepth",
            np.array([0.0, 10.0]),
            np.array([0.1, 0.9]),
        ),
        y_axis_name="ReductionCoeff",
        x_axis_name_default="WaterDepth",
    )
    payload: dict[str, Any] = {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": "Env",
            "elements": {
                "sand": {
                    "accumulationRate": 1.0,
                    "model": {
                        "modelType": "EnvironmentOptimum",
                        "accumulationCurves": [c1, c2],
                    },
                }
            },
        },
    }

    with pytest.raises(
        ValueError, match="Conflicting definitions for accumulation curve"
    ):
        loadAccumulationModelFromJsonObj(payload)


def test_loadAMFromJsonObj_rejects_empty_curve_name_via_loader_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard against loaders returning an curve with empty name."""

    class _FakeCurve:
        _xAxisName = ""
        _abscissa = np.array([0.0, 1.0])
        _ordinate = np.array([0.0, 1.0])

    def _fake_load_inline_or_url(*args: Any, **kwargs: Any) -> list[Any]:
        return [_FakeCurve()]

    monkeypatch.setattr(
        accumulation_model_io,
        "load_inline_or_url",
        _fake_load_inline_or_url,
    )

    payload: dict[str, Any] = {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": "Env",
            "elements": {
                "sand": {
                    "accumulationRate": 1.0,
                    "model": {
                        "modelType": "EnvironmentOptimum",
                        "accumulationCurves": [
                            {"format": "pyWellSFM.CurveData"}
                        ],
                    },
                }
            },
        },
    }

    with pytest.raises(ValueError, match="empty xAxisName"):
        loadAccumulationModelFromJsonObj(payload)


def test_loadAccumulationModelFromJsonObj_url_curve_asserts_single_curve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """URL curve loader currently expects exactly one loaded curve."""

    def _fake_load_curves_from_file(path: Path) -> list[Any]:
        return []

    monkeypatch.setattr(
        accumulation_model_io,
        "loadCurvesFromFile",
        _fake_load_curves_from_file,
    )

    payload: dict[str, Any] = {
        "format": "pyWellSFM.AccumulationModelData",
        "version": "1.0",
        "accumulationModel": {
            "name": "Env",
            "elements": {
                "sand": {
                    "accumulationRate": 1.0,
                    "model": {
                        "modelType": "EnvironmentOptimum",
                        "accumulationCurves": [{"url": "curve.json"}],
                    },
                }
            },
        },
    }

    with pytest.raises(ValueError, match="expected exactly one"):
        loadAccumulationModelFromJsonObj(payload, base_dir=str(tmp_path))


def test_saveAccumulationModelEnvironmentOptimumToJson_accepts_compat_kwargs(
    tmp_path: Path,
) -> None:
    """Compatibility kwargs are accepted when curves_mode is inline."""
    model = AccumulationModel(
        "Env",
        {
            "sand": AccumulationModelElementOptimum(
                "sand",
                10.0,
                accumulationCurves={
                    "WaterDepth": AccumulationCurve(
                        "WaterDepth",
                        np.array([0.0, 10.0]),
                        np.array([0.0, 1.0]),
                    )
                },
            )
        },
    )
    out_json = tmp_path / "compat" / "env.json"
    saveAccumulationModelEnvironmentOptimumToJson(
        model,
        str(out_json),
        curves_mode="inline",
        curves_dir="unused",
        curves_format="csv",
    )

    assert out_json.exists()


def test_saveAccumulationModel_json_and_csv_and_bad_extension(
    tmp_path: Path,
) -> None:
    """Generic save routes by extension and rejects unsupported outputs."""
    model = AccumulationModel(
        "G",
        {
            "sand": AccumulationModelElementGaussian("sand", 10.0, 0.2),
            "shale": AccumulationModelElementGaussian("shale", 4.0, 0.1),
        },
    )

    out_json = tmp_path / "generic" / "accumulation.json"
    saveAccumulationModel(model, str(out_json))
    assert out_json.exists()
    reloaded_json = loadAccumulationModel(str(out_json))
    assert _gaussian_signature(reloaded_json) == _gaussian_signature(model)

    out_csv = tmp_path / "generic" / "accumulation.csv"
    saveAccumulationModel(model, str(out_csv))
    assert out_csv.exists()
    reloaded_csv = loadAccumulationModel(str(out_csv))
    assert _gaussian_signature(reloaded_csv) == _gaussian_signature(model)

    with pytest.raises(
        ValueError, match="Unsupported accumulation model output extension"
    ):
        saveAccumulationModel(
            model, str(tmp_path / "generic" / "accumulation.bin")
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
