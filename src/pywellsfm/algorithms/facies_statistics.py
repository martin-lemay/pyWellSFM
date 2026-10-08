# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

"""Statistics of discrete (facies or environment) successions.

This module provides tools to characterize vertical successions of discrete
states (facies, depositional environments, lithologies) read from a
:class:`striplog.Striplog`:

- extraction of the stratigraphic sequence of states and bed thicknesses,
- transition count and probability matrices (embedded or not),
- chi-square test of the Markov property of an embedded chain against the
  quasi-independence model (Goodman, 1968; Powers and Easterling, 1982),
- thickness proportions and bed thickness statistics.

Embedded chains only record changes of state, so the diagonal of the
transition count matrix is structurally zero. The null hypothesis of
independence must then be expressed by the quasi-independence model, whose
expected counts are obtained by iterative proportional fitting
(Powers and Easterling, 1982).

References:
- Goodman, L. A. (1968). The analysis of cross-classified data:
  independence, quasi-independence, and interactions in contingency tables
  with or without missing entries. J. Am. Stat. Assoc., 63, 1091-1131.
- Powers, D. W., and Easterling, R. G. (1982). Improved methodology for
  using embedded Markov chains to describe cyclical sediments. J. Sediment.
  Petrol., 52, 913-923.
- Burgess, P. M., and Pollitt, D. A. (2012). The origins of shallow-water
  carbonate lithofacies thickness distributions: one-dimensional forward
  modelling of relative sea-level and production rate control.
  Sedimentology, 59, 57-80.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from striplog import Interval, Striplog

__all__ = [
    "MarkovTestResult",
    "bedThicknessStatistics",
    "embeddedMarkovChiSquareTest",
    "pooledTransitionCountMatrix",
    "quasiIndependenceExpectedCounts",
    "striplogToSequence",
    "thicknessProportionDistance",
    "thicknessProportions",
    "transitionCountMatrix",
    "transitionProbabilityMatrix",
]


def _intervalLabel(interval: Interval, component: str) -> str:
    """Get the label of the primary component of an interval.

    :param Interval interval: striplog interval.
    :param str component: name of the component attribute.
    :return str: label, or an empty string if undefined.
    """
    primary = interval.primary
    if primary is None:
        return ""
    try:
        value = primary[component]
    except (KeyError, AttributeError):
        return ""
    return "" if value is None else str(value)


def striplogToSequence(
    log: Striplog,
    component: str = "lithology",
    order: Literal["upward", "downward"] = "upward",
    merge: bool = True,
    dropZeroThickness: bool = True,
) -> tuple[list[str], npt.NDArray[np.float64]]:
    """Extract the sequence of states and bed thicknesses from a striplog.

    Depth increases downward in a striplog, so the stratigraphic (upward)
    order is from the deepest interval to the shallowest one.

    :param Striplog log: discrete log in depth domain.
    :param str component: name of the component attribute holding the
        state label. Default is "lithology".
    :param Literal["upward", "downward"] order: "upward" returns the
        sequence from base to top (stratigraphic order), "downward" from top
        to base. Default is "upward".
    :param bool merge: if True, adjacent intervals with the same label are
        merged into a single bed. Default is True.
    :param bool dropZeroThickness: if True, intervals of zero thickness
        (e.g., environments recorded during non-deposition or exposure in
        simulated wells) are ignored before merging. Default is True.
    :return tuple[list[str], npt.NDArray[np.float64]]: labels and bed
        thicknesses in the requested order.
    """
    if order not in ("upward", "downward"):
        raise ValueError("order must be 'upward' or 'downward'.")
    intervals = sorted(log, key=lambda iv: iv.top.z)
    labels: list[str] = []
    thicknesses: list[float] = []
    for interval in intervals:
        label = _intervalLabel(interval, component)
        thickness = float(abs(interval.base.z - interval.top.z))
        if dropZeroThickness and thickness <= 0.0:
            continue
        if merge and labels and labels[-1] == label:
            thicknesses[-1] += thickness
        else:
            labels.append(label)
            thicknesses.append(thickness)
    if order == "upward":
        labels = labels[::-1]
        thicknesses = thicknesses[::-1]
    return labels, np.asarray(thicknesses, dtype=np.float64)


def _collapseRepeats(sequence: Sequence[str]) -> list[str]:
    """Remove consecutive repetitions of the same state.

    :param Sequence[str] sequence: sequence of states.
    :return list[str]: sequence where consecutive states differ.
    """
    collapsed: list[str] = []
    for state in sequence:
        if not collapsed or collapsed[-1] != state:
            collapsed.append(state)
    return collapsed


def transitionCountMatrix(
    sequence: Sequence[str],
    states: Sequence[str] | None = None,
    embedded: bool = True,
) -> tuple[npt.NDArray[np.float64], list[str]]:
    """Count transitions between successive states of a sequence.

    :param Sequence[str] sequence: ordered sequence of states (e.g., from
        base to top).
    :param Sequence[str] | None states: ordered list of states defining the
        rows and columns of the matrix. If None, sorted unique states of the
        sequence are used. States of the sequence that are not in this list
        are ignored.
    :param bool embedded: if True, consecutive repetitions are collapsed so
        that only changes of state are counted (embedded Markov chain) and
        the diagonal is zero. Default is True.
    :return tuple[npt.NDArray[np.float64], list[str]]: count matrix where
        element (i, j) is the number of transitions from state i to state j,
        and the list of states.
    """
    seq = list(sequence)
    if embedded:
        seq = _collapseRepeats(seq)
    stateList = sorted(set(seq)) if states is None else list(states)
    index = {state: i for i, state in enumerate(stateList)}
    counts = np.zeros((len(stateList), len(stateList)), dtype=np.float64)
    for src, dst in zip(seq[:-1], seq[1:], strict=True):
        if src in index and dst in index:
            counts[index[src], index[dst]] += 1.0
    return counts, stateList


def pooledTransitionCountMatrix(
    sequences: Sequence[Sequence[str]],
    states: Sequence[str] | None = None,
    embedded: bool = True,
) -> tuple[npt.NDArray[np.float64], list[str]]:
    """Sum transition counts over several sequences (e.g., realizations).

    Transitions are never counted across two sequences.

    :param Sequence[Sequence[str]] sequences: list of state sequences.
    :param Sequence[str] | None states: ordered list of states. If None,
        sorted unique states over all sequences are used.
    :param bool embedded: if True, count embedded transitions only.
    :return tuple[npt.NDArray[np.float64], list[str]]: pooled count matrix
        and list of states.
    """
    if states is None:
        states = sorted({state for seq in sequences for state in seq})
    stateList = list(states)
    total = np.zeros((len(stateList), len(stateList)), dtype=np.float64)
    for seq in sequences:
        counts, _ = transitionCountMatrix(seq, stateList, embedded)
        total += counts
    return total, stateList


def transitionProbabilityMatrix(
    counts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Normalize a transition count matrix by rows.

    :param npt.NDArray[np.float64] counts: transition count matrix.
    :return npt.NDArray[np.float64]: transition probability matrix. Rows of
        states that are never left are filled with NaN.
    """
    counts = np.asarray(counts, dtype=np.float64)
    rowSums = counts.sum(axis=1, keepdims=True)
    probs = np.full_like(counts, np.nan)
    np.divide(counts, rowSums, out=probs, where=rowSums > 0)
    return probs


def quasiIndependenceExpectedCounts(
    counts: npt.NDArray[np.float64],
    maxIter: int = 10000,
    tol: float = 1e-9,
) -> npt.NDArray[np.float64]:
    """Expected counts of an embedded chain under quasi-independence.

    Expected counts e_ij = a_i * b_j for i != j and 0 on the diagonal are
    fitted by iterative proportional fitting so that row and column sums
    match the observed ones (Powers and Easterling, 1982).

    :param npt.NDArray[np.float64] counts: observed embedded transition
        count matrix (zero diagonal).
    :param int maxIter: maximum number of fitting iterations.
    :param float tol: convergence tolerance on margins.
    :return npt.NDArray[np.float64]: expected count matrix.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.shape[0]
    if counts.shape != (n, n):
        raise ValueError("counts must be a square matrix.")
    mask = 1.0 - np.eye(n)
    rowTarget = counts.sum(axis=1)
    colTarget = counts.sum(axis=0)
    expected = mask.copy()
    for _ in range(maxIter):
        rowSums = expected.sum(axis=1)
        factor = np.divide(
            rowTarget,
            rowSums,
            out=np.zeros_like(rowTarget),
            where=rowSums > 0,
        )
        expected *= factor[:, None]
        colSums = expected.sum(axis=0)
        factor = np.divide(
            colTarget,
            colSums,
            out=np.zeros_like(colTarget),
            where=colSums > 0,
        )
        expected *= factor[None, :]
        if np.allclose(
            expected.sum(axis=1), rowTarget, rtol=0.0, atol=tol
        ) and np.allclose(expected.sum(axis=0), colTarget, rtol=0.0, atol=tol):
            break
    return expected


@dataclass(frozen=True)
class MarkovTestResult:
    """Result of the chi-square test of an embedded Markov chain.

    :param float statistic: chi-square statistic.
    :param int dof: degrees of freedom.
    :param float pValue: probability to observe a statistic at least as
        large under the quasi-independence (random ordering) hypothesis.
    :param npt.NDArray[np.float64] expected: expected count matrix.
    """

    statistic: float
    dof: int
    pValue: float
    expected: npt.NDArray[np.float64]


def embeddedMarkovChiSquareTest(
    counts: npt.NDArray[np.float64],
) -> MarkovTestResult:
    """Test the Markov property of an embedded transition count matrix.

    The null hypothesis is that successive states are quasi-independent,
    i.e., the vertical ordering is random apart from the impossibility of
    self-transitions. A small p-value indicates a preferred ordering of
    states (e.g., shallowing-upward cycles).

    States that are never entered or left are dropped. The number of
    degrees of freedom is (n - 1)^2 - n for n remaining states.

    :param npt.NDArray[np.float64] counts: embedded transition count matrix.
    :return MarkovTestResult: test result.
    """
    counts = np.asarray(counts, dtype=np.float64)
    keep = (counts.sum(axis=0) + counts.sum(axis=1)) > 0
    sub = counts[np.ix_(keep, keep)]
    n = sub.shape[0]
    dof = (n - 1) ** 2 - n
    expected = quasiIndependenceExpectedCounts(sub)
    if dof <= 0:
        return MarkovTestResult(float("nan"), dof, float("nan"), expected)
    offDiagonal = ~np.eye(n, dtype=bool) & (expected > 0)
    statistic = float(
        np.sum(
            (sub[offDiagonal] - expected[offDiagonal]) ** 2
            / expected[offDiagonal]
        )
    )
    pValue = float(stats.chi2.sf(statistic, dof))
    return MarkovTestResult(statistic, dof, pValue, expected)


def thicknessProportions(
    labels: Sequence[str],
    thicknesses: npt.NDArray[np.float64],
) -> dict[str, float]:
    """Compute the thickness proportion of each state.

    :param Sequence[str] labels: bed labels.
    :param npt.NDArray[np.float64] thicknesses: bed thicknesses.
    :return dict[str, float]: proportion of the total thickness per state.
    """
    thicknesses = np.asarray(thicknesses, dtype=np.float64)
    if len(labels) != thicknesses.size:
        raise ValueError("labels and thicknesses must have the same size.")
    total = float(thicknesses.sum())
    proportions: dict[str, float] = {}
    for label, thickness in zip(labels, thicknesses, strict=True):
        proportions[label] = proportions.get(label, 0.0) + float(thickness)
    if total > 0:
        proportions = {k: v / total for k, v in proportions.items()}
    return dict(sorted(proportions.items()))


def thicknessProportionDistance(
    proportionsA: dict[str, float],
    proportionsB: dict[str, float],
) -> float:
    """Total variation distance between two sets of thickness proportions.

    The distance is half the sum of absolute differences of proportions over
    all states. It ranges from 0 (same proportions) to 1 (no state in
    common), and ignores the vertical position of beds.

    :param dict[str, float] proportionsA: proportions per state.
    :param dict[str, float] proportionsB: proportions per state.
    :return float: total variation distance.
    """
    states = set(proportionsA) | set(proportionsB)
    return 0.5 * float(
        sum(
            abs(proportionsA.get(state, 0.0) - proportionsB.get(state, 0.0))
            for state in states
        )
    )


def bedThicknessStatistics(
    labels: Sequence[str],
    thicknesses: npt.NDArray[np.float64],
) -> pd.DataFrame:
    """Summarize bed thicknesses per state.

    :param Sequence[str] labels: bed labels (adjacent identical labels are
        expected to be merged beforehand).
    :param npt.NDArray[np.float64] thicknesses: bed thicknesses.
    :return pd.DataFrame: one row per state with columns count, mean, std,
        median, min, max and total thickness.
    """
    frame = pd.DataFrame(
        {"state": list(labels), "thickness": np.asarray(thicknesses)}
    )
    if frame.empty:
        return pd.DataFrame(
            columns=["count", "mean", "std", "median", "min", "max", "total"]
        )
    grouped = frame.groupby("state")["thickness"]
    return pd.DataFrame(
        {
            "count": grouped.count(),
            "mean": grouped.mean(),
            "std": grouped.std(ddof=1),
            "median": grouped.median(),
            "min": grouped.min(),
            "max": grouped.max(),
            "total": grouped.sum(),
        }
    )
