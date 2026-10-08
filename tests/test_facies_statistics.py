# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file
import os
import sys

m_path = os.path.dirname(os.getcwd())
if m_path not in sys.path:
    sys.path.insert(0, os.path.join(m_path, "src"))

import numpy as np
import pytest
from striplog import Component, Interval, Striplog

from pywellsfm.algorithms import (
    bedThicknessStatistics,
    embeddedMarkovChiSquareTest,
    pooledTransitionCountMatrix,
    quasiIndependenceExpectedCounts,
    striplogToSequence,
    thicknessProportionDistance,
    thicknessProportions,
    transitionCountMatrix,
    transitionProbabilityMatrix,
)


def _striplog(rows: list[tuple[float, float, str]]) -> Striplog:
    """Build a striplog from (top, base, label) rows."""
    return Striplog(
        [
            Interval(top, base, components=[Component({"lithology": label})])
            for top, base, label in rows
        ]
    )


@pytest.fixture
def log() -> Striplog:
    """Striplog with a repeated label to merge (depth increases downward)."""
    return _striplog(
        [
            (0.0, 1.0, "a"),
            (1.0, 2.5, "b"),
            (2.5, 3.0, "b"),
            (3.0, 4.0, "c"),
            (4.0, 5.0, "a"),
        ]
    )


def test_striplog_to_sequence_upward_merges_beds(log: Striplog) -> None:
    """Sequence is read from base to top and adjacent beds are merged."""
    labels, thicknesses = striplogToSequence(log)
    assert labels == ["a", "c", "b", "a"]
    np.testing.assert_allclose(thicknesses, [1.0, 1.0, 2.0, 1.0])


def test_striplog_to_sequence_downward_without_merge(log: Striplog) -> None:
    """Downward order keeps the depth order and all intervals."""
    labels, thicknesses = striplogToSequence(
        log, order="downward", merge=False
    )
    assert labels == ["a", "b", "b", "c", "a"]
    np.testing.assert_allclose(thicknesses, [1.0, 1.5, 0.5, 1.0, 1.0])


def test_striplog_to_sequence_invalid_order(log: Striplog) -> None:
    """Invalid order raises a ValueError."""
    with pytest.raises(ValueError):
        striplogToSequence(log, order="sideways")  # type: ignore[arg-type]


def test_transition_count_matrix_embedded() -> None:
    """Embedded counts ignore repetitions and have a zero diagonal."""
    counts, states = transitionCountMatrix(["a", "a", "b", "c", "a", "b"])
    assert states == ["a", "b", "c"]
    expected = np.array([[0, 2, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    np.testing.assert_array_equal(counts, expected)


def test_transition_count_matrix_not_embedded_with_states() -> None:
    """Non-embedded counts keep self-transitions; unknown states ignored."""
    counts, states = transitionCountMatrix(
        ["a", "a", "b", "x", "b"], states=["b", "a"], embedded=False
    )
    assert states == ["b", "a"]
    expected = np.array([[0, 0], [1, 1]], dtype=float)
    np.testing.assert_array_equal(counts, expected)


def test_pooled_counts_do_not_link_sequences() -> None:
    """No transition is counted between the end and start of sequences."""
    counts, states = pooledTransitionCountMatrix([["a", "b"], ["b", "a"]])
    assert states == ["a", "b"]
    np.testing.assert_array_equal(counts, [[0, 1], [1, 0]])


def test_transition_probability_matrix_rows_sum_to_one() -> None:
    """Rows are normalized and empty rows are NaN."""
    probs = transitionProbabilityMatrix(
        np.array([[0, 2, 2], [1, 0, 0], [0, 0, 0]], dtype=float)
    )
    np.testing.assert_allclose(probs[0], [0, 0.5, 0.5])
    np.testing.assert_allclose(probs[1], [1, 0, 0])
    assert np.all(np.isnan(probs[2]))


def test_quasi_independence_matches_margins() -> None:
    """Expected counts reproduce margins with a zero diagonal."""
    counts = np.array(
        [[0, 10, 2, 3], [4, 0, 9, 1], [2, 3, 0, 8], [9, 1, 2, 0]], dtype=float
    )
    expected = quasiIndependenceExpectedCounts(counts)
    np.testing.assert_allclose(np.diag(expected), 0.0)
    np.testing.assert_allclose(expected.sum(axis=1), counts.sum(axis=1))
    np.testing.assert_allclose(expected.sum(axis=0), counts.sum(axis=0))


def test_quasi_independence_requires_square_matrix() -> None:
    """Non square matrices are rejected."""
    with pytest.raises(ValueError):
        quasiIndependenceExpectedCounts(np.zeros((2, 3)))


def test_markov_test_detects_cyclic_ordering() -> None:
    """A strictly cyclic succession rejects quasi-independence."""
    counts, _ = transitionCountMatrix(list("abcd" * 30))
    result = embeddedMarkovChiSquareTest(counts)
    assert result.dof == 5
    assert result.pValue < 1e-6


def test_markov_test_accepts_random_ordering() -> None:
    """A random succession does not reject quasi-independence."""
    rng = np.random.default_rng(0)
    seq: list[str] = ["a"]
    states = ["a", "b", "c", "d"]
    while len(seq) < 2000:
        nxt = str(rng.choice(states))
        if nxt != seq[-1]:
            seq.append(nxt)
    counts, _ = transitionCountMatrix(seq)
    result = embeddedMarkovChiSquareTest(counts)
    assert result.pValue > 0.01


def test_markov_test_undefined_for_two_states() -> None:
    """Two states give no degree of freedom and a NaN statistic."""
    counts, _ = transitionCountMatrix(list("ab" * 5))
    result = embeddedMarkovChiSquareTest(counts)
    assert result.dof <= 0
    assert np.isnan(result.pValue)


def test_thickness_proportions() -> None:
    """Proportions sum to one and are grouped by label."""
    props = thicknessProportions(["a", "b", "a"], np.array([1.0, 2.0, 1.0]))
    assert props == {"a": 0.5, "b": 0.5}
    with pytest.raises(ValueError):
        thicknessProportions(["a"], np.array([1.0, 2.0]))


def test_bed_thickness_statistics() -> None:
    """Statistics are computed per state."""
    table = bedThicknessStatistics(["a", "b", "a"], np.array([1.0, 2.0, 3.0]))
    assert table.loc["a", "count"] == 2
    assert table.loc["a", "mean"] == pytest.approx(2.0)
    assert table.loc["b", "total"] == pytest.approx(2.0)
    assert bedThicknessStatistics([], np.array([])).empty


def test_thickness_proportion_distance() -> None:
    """Total variation distance between proportions."""
    assert thicknessProportionDistance({"a": 1.0}, {"a": 1.0}) == 0.0
    assert thicknessProportionDistance({"a": 1.0}, {"b": 1.0}) == 1.0
    assert thicknessProportionDistance(
        {"a": 0.5, "b": 0.5}, {"a": 0.25, "b": 0.75}
    ) == pytest.approx(0.25)


def test_striplog_to_sequence_zero_thickness() -> None:
    """Zero-thickness intervals are dropped by default, then beds merged."""
    log = _striplog([(0.0, 1.0, "a"), (1.0, 1.0, "b"), (1.0, 2.0, "a")])
    labels, thicknesses = striplogToSequence(log)
    assert labels == ["a"]
    np.testing.assert_allclose(thicknesses, [2.0])
    labels, _ = striplogToSequence(log, dropZeroThickness=False)
    assert labels == ["a", "b", "a"]
