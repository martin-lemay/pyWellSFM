# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

"""Checks on the example notebooks rendered in the documentation."""

import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

# Absolute paths of a local machine: Windows drive paths, user/system
# directories of POSIX systems and installed packages.
ABSOLUTE_PATH = re.compile(
    r"\b[A-Za-z]:[\\/]+[^\\/\s\"'<>]+[\\/]"
    r"|(?<![\w.])/(?:home|Users|root|mnt|tmp|opt|var)/"
    r"|site-packages"
)


def _cell_texts(cell: dict) -> Iterator[tuple[str, str]]:
    """Yield (location, text) of the source and text outputs of a cell."""
    yield "source", "".join(cell["source"])
    for output in cell.get("outputs", []):
        kind = output["output_type"]
        if "text" in output:
            yield kind, "".join(output["text"])
        if "traceback" in output:
            yield kind, "\n".join(output["traceback"])
        for mime, data in output.get("data", {}).items():
            if mime.startswith("text/"):
                text = "".join(data) if isinstance(data, list) else data
                yield f"{kind} {mime}", text


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_absolute_path(notebook: Path) -> None:
    """Notebooks are published in the docs: no local path must leak."""
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    leaks = [
        f"cell {index} ({location}): {match.group(0)!r}"
        for index, cell in enumerate(cells)
        for location, text in _cell_texts(cell)
        for match in ABSOLUTE_PATH.finditer(text)
    ]
    assert not leaks, "Absolute paths found:\n" + "\n".join(leaks)
