# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from __future__ import annotations

from pathlib import Path

import pytest
from striplog import Striplog

from pywellsfm.io.striplog_io import _importStriplogFromCsv, importStriplog


def test_importStriplog_csv_loads_and_converts_depth_to_cm(
    tmp_path: Path,
) -> None:
    """CSV import scales top/base values from m to cm."""
    csv_path = tmp_path / "litho.csv"
    csv_path.write_text(
        "top,base,lithology\n1.0,1.5,sandstone\n1.5,2.0,shale\n",
        encoding="utf-8",
    )

    striplog = importStriplog(str(csv_path))
    assert isinstance(striplog, Striplog)
    assert len(striplog) == 2
    assert striplog[0].top.middle == 100.0
    assert striplog[0].base.middle == 150.0
    assert striplog[1].top.middle == 150.0
    assert striplog[1].base.middle == 200.0
    assert striplog[0].components[0]["lithology"] == "sandstone"
    assert striplog[1].components[0]["lithology"] == "shale"


def test_importStriplog_csv_supports_custom_delimiter(tmp_path: Path) -> None:
    """Delimiter argument is passed to pandas CSV reader."""
    csv_path = tmp_path / "litho_semicolon.csv"
    csv_path.write_text(
        "top;base;lithology\n0.0;1.0;limestone\n",
        encoding="utf-8",
    )

    striplog = importStriplog(str(csv_path), csvDelimiter=";")
    assert isinstance(striplog, Striplog)
    assert len(striplog) == 1
    assert striplog[0].top.middle == 0.0
    assert striplog[0].base.middle == 100.0
    assert striplog[0].components[0]["lithology"] == "limestone"


def test_importStriplog_rejects_missing_file_and_unsupported_extension(
    tmp_path: Path,
) -> None:
    """Public striplog import validates path existence and extension."""
    with pytest.raises(FileNotFoundError):
        importStriplog(str(tmp_path / "missing.csv"))

    txt_path = tmp_path / "litho.txt"
    txt_path.write_text("dummy", encoding="utf-8")
    with pytest.raises(ValueError):
        importStriplog(str(txt_path))


def test_importStriplogFromCsv_preserves_non_depth_columns(
    tmp_path: Path,
) -> None:
    """All non top/base columns are preserved in component metadata."""
    csv_path = tmp_path / "litho_with_props.csv"
    csv_path.write_text(
        (
            "top,base,lithology,color,porosity\n"
            "0.0,0.5,sandstone,yellow,0.25\n"
        ),
        encoding="utf-8",
    )

    striplog = _importStriplogFromCsv(csv_path, delimiter=",")
    assert isinstance(striplog, Striplog)
    assert len(striplog) == 1
    comp = striplog[0].components[0]
    assert comp["lithology"] == "sandstone"
    assert comp["color"] == "yellow"
    assert comp["porosity"] == pytest.approx(0.25)
