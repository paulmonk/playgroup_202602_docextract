"""Value normalisation functions for money and date fields."""

from __future__ import annotations

import datetime as dt
import logging

logger = logging.getLogger(__name__)


def normalise_money(value: str) -> str:
    """Strip currency symbols and commas, format to 2 decimal places.

    Args:
        value: Raw money string (may contain pounds sign, commas, whitespace).

    Returns:
        Formatted string like ``255653.00``, or the cleaned input if not numeric.
    """
    cleaned = value.replace(",", "").replace("\u00a3", "").strip()
    try:
        return f"{float(cleaned):.2f}"
    except (ValueError, TypeError):
        logger.debug("Money normalisation fallback for non-numeric value: %r", value)
        return cleaned


def normalise_date(value: str) -> str:
    """Parse common date formats to ISO 8601 (YYYY-MM-DD).

    Args:
        value: Date string in various formats.

    Returns:
        ISO date string like ``2015-12-31``, or the input unchanged if unparseable.
    """
    try:
        return dt.date.fromisoformat(value).isoformat()
    except (ValueError, TypeError):
        pass

    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
        try:
            return dt.datetime.strptime(value, fmt).replace(tzinfo=dt.UTC).date().isoformat()
        except (ValueError, TypeError):
            continue

    logger.debug("Date normalisation fallback for unrecognised format: %r", value)
    return value
