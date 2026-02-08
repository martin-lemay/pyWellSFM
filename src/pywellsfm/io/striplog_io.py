# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from pathlib import Path

import pandas as pd
from striplog import Component, Interval, Striplog


def importStriplog(filepath: str, csvDelimiter: str = ",") -> Striplog:
    """Load a stripLog from a file.

    Supported file format are:
        - .csv : comma-separated values file

    :param str filepath: Path to the striplog CSV file.
    :return Striplog: Loaded striplog object.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(str(path))

    ext = path.suffix.lower()
    if ext == ".csv":
        return _importStriplogFromCsv(path, csvDelimiter)

    raise ValueError("Unsupported file format for striplog import: " + ext)


def _importStriplogFromCsv(path: Path, delimiter: str) -> Striplog:
    """Load a stripLog from a csv file.

    csv file contains a row per layer, each layer has "top", "base",

    :param Path path: Path to the striplog CSV file.
    :return Striplog: Loaded striplog object.
    """
    df = pd.read_csv(str(path), delimiter=delimiter, engine="python")
    # Convert from m to cm
    df.loc[:, ("top", "base")] *= 100  # type: ignore
    intervals = []
    for _, row in df.iterrows():
        compDict = {key: row[key] for key in row.index if key not in ("top", "base")}
        interval = Interval(
            row["top"],
            row["base"],
            components=[Component(compDict)],
        )
        intervals.append(interval)
    return Striplog(intervals)
