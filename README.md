[![CI](https://github.com/martin-lemay/pyWellSFM/actions/workflows/python-package.yml/badge.svg)](https://github.com/martin-lemay/pyWellSFM/actions)
[![docs](https://readthedocs.org/projects/pywellsfm/badge/?version=latest)](https://pywellsfm.readthedocs.io/en/latest/)

# Welcome to pyWellSFM Repo!
pyWellSFM stands for Python Well Stratigraphic Forward Modeling.

This package aims at simulating the deposition of sedimentary layers over time along one or multiple well(s). The deposition of elements is controlled by the accommodation space -split into eustatism variations and subsidence- and the accumulation model. The accumulation model set the rules for the elements to be accumulated. Two models are currently implemented:

- a Gaussian model: at each step, the accumulated thickness of each element follows a Normal law
- a Environment Optimum model: at each step, the accumulated thickness of each element depends on environment conditions. When conditions are optimal, the rate is maximal, but it decreases according to the acummulation curves when conditions move away from optimum values.

Time step duration is computed such as both deposited thickness and water depth variation do not exceed a user-defined value (0.5m by default).

The simulator is designed such as it can easily be used through an optimization loop.

A full documentation of the code can be found [here](https://pywellsfm.readthedocs.io/en/latest/)


## Quickstart

Install from GitHub:

```bash
pip install git+https://github.com/martin-lemay/pyWellSFM.git
```

Minimal example, run from the `tests/data/` folder of this repository
(`simulation.json` references `eustatic_curve.csv` by relative path):

```python
from pywellsfm.io import loadFSSimulation
from pywellsfm.utils import plot_litho_log

# 1) Load the simulation data (scenario + realizations) from a JSON file
simulator = loadFSSimulation("simulation.json")

# 2) Run the simulation from the oldest marker up to the top of the well
simulator.prepare()
simulator.run()
simulator.finalize()

# 3) Inspect results: an xarray Dataset with one entry per realization ...
print(simulator.outputs)

# ... and one simulated well per realization, with lithology logs
well = simulator.simulatedWells[0]
print(well.getDiscreteLogNames())

# 4) Plot the simulated lithology log of the first realization
fig = plot_litho_log(well, "MainElement")
fig.show()
```

By default, warnings and errors are printed to the console and all messages
from INFO up are kept in memory (`pywellsfm.get_stored_logs()`). Call
`pywellsfm.configure_logging(level=pywellsfm.INFO)` to also print progress
messages.

Supported input formats:

- wells (use `loadWell()`):
  - LAS 2.0
  - json: see json schema in https://raw.githubusercontent.com/martin-lemay/pyWellSFM/main/src/pywellsfm/jsonSchemas/WellSchema.json

- curves (subsidence, eustatism, accumulation curve, etc.; use `loadCurvesFromFile()`):
  - csv: expects 2 columns, `AbscissaName` (e.g., "Age", "WaterDepth") and `CurveName` (e.g., "Eustacy", "Subsidence", "ReductionCoeff").
  - json: see json schema in https://raw.githubusercontent.com/martin-lemay/pyWellSFM/main/src/pywellsfm/jsonSchemas/CurveSchema.json

- Accumulation model (use `loadAccumulationModel()`):
  - json: see json schema in https://raw.githubusercontent.com/martin-lemay/pyWellSFM/main/src/pywellsfm/jsonSchemas/AccumulationModelSchema.json

- Facies model (use `loadFaciesModel()`):
  - json: see json schema in https://raw.githubusercontent.com/martin-lemay/pyWellSFM/main/src/pywellsfm/jsonSchemas/FaciesModelSchema.json

- Scenario (use `loadScenario()`):
  - json: see json schema in https://raw.githubusercontent.com/martin-lemay/pyWellSFM/main/src/pywellsfm/jsonSchemas/ScenarioSchema.json

- Simulation data (use `loadFSSimulation()`):
  - json: see json schema in https://raw.githubusercontent.com/martin-lemay/pyWellSFM/main/src/pywellsfm/jsonSchemas/FSSimulationDataSchema.json

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

# Coverage gate (configured in pyproject.toml)
# Test run fails if total coverage is below 80%
pytest --cov=pywellsfm --cov-fail-under=80

```

### Build the docs locally

```bash
python -m sphinx -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in your browser.

### What is checked in CI

On each Pull Request, GitHub Actions runs:

- `ruff format --check` and `ruff check` (formatting and lint)
- `mypy` (static type checks)
- package build, then a smoke test of the installed wheel
- `pytest` (unit tests)
- Coverage threshold: test run fails if total coverage is below 80%

## Credits
pyWellSFM was written by [Martin Lemay](https://github.com/martin-lemay) <br>[![ORCID Badge](https://img.shields.io/badge/ORCID-A6CE39?logo=orcid&logoColor=fff&style=flat-square)](https://orcid.org/0000-0002-5538-7885)</br>

## Citation
If you use pyWellSFM in your work, please cite it. Citation metadata is in
[CITATION.cff](CITATION.cff); on GitHub, use the "Cite this repository" button.
Each release is archived on Zenodo with a DOI.

## License
pyWellSFM is licensed under [Apache-2.0 license](https://opensource.org/licenses/Apache-2.0).
