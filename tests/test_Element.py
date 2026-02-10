# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay
# ruff: noqa: E402 # disable Module level import not at top of file

"""Unit tests for Element."""

import os
import sys

import pytest

m_path = os.path.join(os.path.dirname(os.getcwd()), "src")
if m_path not in sys.path:
    sys.path.insert(0, m_path)


from pywellsfm.model import Element


# Parameterize the tests with element names and production max values
@pytest.mark.parametrize(
    "element_name, accumulation_rate",
    [("Sand", 10), ("Shale", 30), ("Carbonate", 50)],
)
def test_element_initialization(
    element_name: str, accumulation_rate: float
) -> None:
    """Test of element initialization.

    :param str element_name: element name
    :param float accumulation_rate: accumulation rate
    """
    element = Element(element_name, accumulation_rate)
    assert element.name == element_name
    assert element.accumulationRate == accumulation_rate


@pytest.mark.parametrize("element_name", ["Sand", "Shale", "Carbonate"])
def test_element_repr1(element_name: str) -> None:
    """Test element representation.

    :param str element_name: element name
    """
    element = Element(element_name, 10.0)
    assert repr(element) == f"{element_name}"


@pytest.mark.parametrize("element_name", ["Sand", "Shale", "Carbonate"])
def test_element_hash(element_name: str) -> None:
    """Test element hashing based on name.

    :param str element_name: element name
    """
    element1 = Element(element_name, 10.0)
    element2 = Element(
        element_name, 5.0
    )  # Same name, different accumulationRate
    element3 = Element("Clay", 10.0)  # Different name

    assert hash(element1) == hash(element2)  # Same name, same hash
    assert hash(element1) != hash(element3)  # Different name, different hash


@pytest.mark.parametrize("element_name", ["Sand", "Shale", "Carbonate"])
def test_element_equality(element_name: str) -> None:
    """Test element equality based on name.

    :param str element_name: element name
    """
    element1 = Element(element_name, 10.0)
    element2 = Element(element_name, 5.0)  # Same name
    element3 = Element("Clay", 10.0)  # Different name

    assert element1 == element2  # Same name
    assert element1 != element3  # Different name


def test_set_Element() -> None:
    """Test usage of Element in a set."""
    element1 = Element("Sand", 10.0)
    element2 = Element("Shale", 30.0)
    element3 = Element("Carbonate", 50.0)
    element4 = Element("Sand", 50.0)

    elementSet = {element1, element2, element3}
    assert len(elementSet) == 3, "The length of the set of elements is wrong."

    elementSet.add(element4)
    assert len(elementSet) == 3, (
        "The length of the set of elements after addition of "
        "element4 is wrong."
    )


if __name__ == "__main__":
    pytest.main(
        [
            os.path.abspath(__file__),
        ]
    )
