"""Shared utilities for text extraction and experiment management.

Re-exports normalisation functions and table formatting so existing
``from lib.utils import ...`` imports continue to work.
"""

from __future__ import annotations

from lib.utils.normalise import normalise_date, normalise_money
from lib.utils.table import format_table

__all__ = ["format_table", "normalise_date", "normalise_money"]
