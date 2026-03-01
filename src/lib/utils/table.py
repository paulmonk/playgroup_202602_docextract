"""Text table formatting."""

from __future__ import annotations


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
