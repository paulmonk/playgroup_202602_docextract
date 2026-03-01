"""Field-type constants shared across extraction and scoring."""

from __future__ import annotations

from typing import Final

# Fields where values are monetary amounts (normalised to 2dp)
MONEY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "income_annually_in_british_pounds",
        "spending_annually_in_british_pounds",
    }
)

# Fields containing dates (normalised to ISO 8601)
DATE_FIELDS: Final[frozenset[str]] = frozenset({"report_date"})

# Fields that should be uppercased in Kleister output
UPPERCASE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "address__post_town",
        "address__postcode",
    }
)
