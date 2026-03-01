"""Tests for value normalisation functions."""

from __future__ import annotations

import pytest

from lib.utils.normalise import normalise_date, normalise_money


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("255653", "255653.00"),
        ("255,653", "255653.00"),
        ("255653.00", "255653.00"),
        ("18797665.0", "18797665.00"),
        ("0", "0.00"),
        ("1.5", "1.50"),
    ],
)
def test_normalise_money(value, expected):
    assert normalise_money(value) == expected


def test_normalise_money_strips_pound_sign():
    assert normalise_money("\u00a3255,653") == "255653.00"


def test_normalise_money_strips_whitespace():
    assert normalise_money("  100.5  ") == "100.50"


def test_normalise_money_non_numeric_returns_cleaned():
    assert normalise_money("N/A") == "N/A"


def test_normalise_money_empty_string():
    assert normalise_money("") == ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2015-12-31", "2015-12-31"),
        ("31/12/2015", "2015-12-31"),
        ("31-12-2015", "2015-12-31"),
        ("31.12.2015", "2015-12-31"),
        ("2020-01-01", "2020-01-01"),
    ],
)
def test_normalise_date(value, expected):
    assert normalise_date(value) == expected


def test_normalise_date_unrecognised_returns_input():
    assert normalise_date("Dec 2015") == "Dec 2015"


def test_normalise_date_empty_returns_input():
    assert normalise_date("") == ""
