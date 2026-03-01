"""Shared utilities for text extraction and experiment management.

Centralises field-type constants and normalisation so the extraction
pipeline and the scorer always agree on how values are compared.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Final

logger = logging.getLogger(__name__)

# Fields where values are monetary amounts (normalised to 2dp)
MONEY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "income_annually_in_british_pounds",
        "spending_annually_in_british_pounds",
    }
)

# Fields containing dates (normalised to ISO 8601)
DATE_FIELDS: Final[frozenset[str]] = frozenset({"report_date"})


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


def format_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> list[str]:
    """Format rows as a left-justified, fixed-width text table.

    Args:
        headers: Column header strings.
        rows: Data rows, each a tuple of cell strings matching headers length.

    Returns:
        List of lines: header, separator, then one line per row.
    """
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    return [fmt(headers), "  ".join("-" * w for w in widths), *(fmt(r) for r in rows)]
