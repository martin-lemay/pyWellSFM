[![CI](https://github.com/martin-lemay/pyWellSFM/actions/workflows/python-package.yml/badge.svg)](https://github.com/martin-lemay/pyWellSFM/actions)
[![docs](https://readthedocs.com/projects/mlemay-pywellsfm/badge/?version=latest)](https://pywellsfm.readthedocs.io/en/latest/)

# Welcome to pyWellSFM Repo!
pyWellSFM stands for Python Well Stratigraphic Forward Modeling.

This package aims at simulating the deposition of sedimentary layers over time along one or multiple well(s). The deposition of elements is controlled by the accommodation space -split into eustatism variations and subsidence- and the accumulation model. The accumulation model set the rules for the elements to be accumulated. Two models are currently implemented:

- a Gaussian model: at each step, the accumulated thickness of each element follows a Normal law
- a Environment Optimum model: at each step, the accumulated thickness of each element depends on environment conditions. When conditions are optimal, the rate is maximal, but it decreases according to the acummulation curves when conditions move away from optimum values.

Time step duration is computed such as both deposited thickness and bathymetry variation do not exceed a user-defined value (0.5m by default).

The simulator is designed such as it can easily be used through an optimization loop.

A full documentation of the code can be found [here](https://pywellsfm.readthedocs.io/en/latest/)


## Quickstart

Install from GitHub:

```bash
pip install git+https://github.com/martin-lemay/pyWellSFM.git
```

Minimal example (data loading and simulation run):

```python
from pywellsfm.io import loadSimulation
from pywellsfm import (
    FSSimulator,
    FSSimulatorRunner,
    RealizationData,
    Scenario,
)

# 1) Load simulation data


# 2) Create simulation object


# 3) Run the simulation


# 4) Plot results


```

Supported input formats:

- wells (use `loadWell()`):
  - LAS 2.0
  - json: see json schema in https://raw.githubusercontent.com/martin-lemay/pyWellSFM/main/jsonSchemas/WellSchema.json

- curves (subsidence, eustatism, accumulation curve, etc.; use `loadCurvesFromFile()`):
  - csv: expects 2 columns, `AbscissaName` (e.g., "Age", "Bathymetry") and `CurveName` (e.g., "Eustacy", "Subsidence", "ReductionCoeff").
  - json: see json schema in https://raw.githubusercontent.com/martin-lemay/pyWellSFM/main/jsonSchemas/CurveSchema.json

- Accumulation model (use `loadAccumulationModel()`):
  - json: see json schema in https://raw.githubusercontent.com/martin-lemay/pyWellSFM/main/jsonSchemas/AccumulationModelSchema.json

- Facies model (use `loadFaciesModel()`):
  - json: see json schema in https://raw.githubusercontent.com/martin-lemay/pyWellSFM/main/jsonSchemas/FaciesModelSchema.json

- Realization Data(use `loadRealizationData()`):
  - json: see json schema in https://raw.githubusercontent.com/martin-lemay/pyWellSFM/main/jsonSchemas/RealizationDataSchema.json

- Scenario (use `loadScenario()`):
  - json: see json schema in https://raw.githubusercontent.com/martin-lemay/pyWellSFM/main/jsonSchemas/ScenarioSchema.json

- Simulation data (use `loadSimulationData()`):
  - json: see json schema in https://raw.githubusercontent.com/martin-lemay/pyWellSFM/main/jsonSchemas/SimulationDataSchema.json

Tip: example files are available in `tests/data/` and test files.

## Installation

Requirements:

- Python >= 3.13

Install from GitHub:

```bash
pip install git+https://github.com/martin-lemay/pyWellSFM.git
```

For development and tests:

```bash
pip install -e .[dev,test]
```

## Contributing

Contributions are welcome — bug reports, feature requests, docs improvements, and code changes.

### Workflow (issues + PR/MR)

- Create an **issue** first to describe the bug / enhancement (with minimal reproducible example when relevant).
- Create a **Pull Request / Merge Request** that **addresses one issue**.
  - Reference the issue in the PR description (e.g. `Fixes #123`).
  - Keep changes focused and include tests/docs updates when applicable.

### Local setup

```bash
pip install -e .[dev,test]
```

If you plan to build the docs locally, install the doc build dependencies as well:

```bash
pip install -r requirements.txt
```

### Formatting, linting, typing, tests

Run these from the repository root:

```bash
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
```

### Build the docs locally

```bash
python -m sphinx -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in your browser.

### What is checked in CI

On each Pull Request, GitHub Actions runs:

- `ruff check` (lint; currently non-blocking in CI)
- `mypy` (static type checks)
- `pytest` (tests + doctests) 

## Credits
pyWellSFM was written by [Martin Lemay](https://github.com/martin-lemay) <br>[![ORCID Badge](https://img.shields.io/badge/ORCID-A6CE39?logo=orcid&logoColor=fff&style=flat-square)](https://orcid.org/my-orcid?orcid=0000-0002-5538-7885)</br>

## License
pyWellSFM is licensed under [Apache-2.0 license](https://opensource.org/licenses/Apache-2.0).
