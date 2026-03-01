"""Field-type-aware F1 scorer for document extraction evaluation.

Computes micro-averaged F1 by treating each (key, normalised_value) pair
as a set member per row, following the Kleister benchmark methodology.
Provides per-field breakdown and near-miss diagnostics.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import Levenshtein

from const import DATE_FIELDS, MONEY_FIELDS
from lib import utils

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class FieldCounts:
    """Confusion matrix counts for a single field."""

    tp: int = 0
    fp: int = 0
    fn: int = 0


@dataclass(kw_only=True, frozen=True)
class Mismatch:
    """A mismatch diagnostic for a single field failure."""

    row: int
    key: str
    expected: str
    predicted: str
    distance: int
    pct_diff: str | None = None
    missing: bool = False


@dataclass(kw_only=True, frozen=True)
class Metrics:
    """Precision, recall, and F1 score."""

    precision: float
    recall: float
    f1: float


@dataclass(kw_only=True, frozen=True)
class PerFieldScore:
    """Per-field scoring result."""

    tp: int
    total: int
    f1: float


@dataclass(kw_only=True, frozen=True)
class ScoresResult:
    """Overall and per-field scoring results."""

    f1: float
    precision: float
    recall: float
    matched: int
    total_fields: int
    per_field: dict[str, PerFieldScore]


def _field_metrics(counts: FieldCounts) -> Metrics:
    """Compute precision, recall, and F1 for a single field.

    Args:
        counts: Confusion matrix counts for the field.

    Returns:
        Metrics with precision, recall, and F1.
    """
    tp = counts.tp
    total = tp + counts.fn
    precision = tp / (tp + counts.fp) if (tp + counts.fp) > 0 else 0.0
    recall = tp / total if total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return Metrics(precision=precision, recall=recall, f1=f1)


def get_all_items(path: Path | str) -> list[dict[str, str]]:
    """Read a Kleister-format file into a list of field dicts.

    Args:
        path: Path to a tab-delimited key=value file.

    Returns:
        One dict per row, mapping field names to values.
    """
    items = []
    with Path(path).open() as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row_num, item in enumerate(reader):
            try:
                k_v_pairs = dict(itm.split("=", 1) for itm in item if itm)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed row {row_num} in {path}: expected key=value pairs, got {item!r}"
                ) from exc
            items.append(k_v_pairs)
    return items


def normalise_value(key: str, *, value: str) -> str:
    """Normalise a value based on field type for fair comparison.

    Args:
        key: The field name, used to dispatch normalisation.
        value: The raw field value.

    Returns:
        The normalised value string.
    """
    value = value.strip()
    if key in MONEY_FIELDS:
        return utils.normalise_money(value).upper()
    if key in DATE_FIELDS:
        return utils.normalise_date(value).upper()
    return value.upper()


def _money_pct_diff(expected: str, predicted: str) -> str:
    """Compute percentage difference between two money values.

    Args:
        expected: Expected money string.
        predicted: Predicted money string.

    Returns:
        Percentage difference as a string, or "N/A" if non-numeric.
    """
    try:
        exp_val = float(expected)
        pred_val = float(predicted)
    except ValueError:
        return "N/A"
    if exp_val == 0.0:
        return "N/A" if pred_val == 0.0 else "inf"
    pct = abs(exp_val - pred_val) / abs(exp_val) * 100
    return f"{pct:.1f}%"


def compute_f1(
    expected_items: list[dict[str, str]],
    *,
    predicted_items: list[dict[str, str]],
) -> tuple[dict[str, FieldCounts], list[Mismatch], dict[str, list[int]]]:
    """Compute per-field true positives, false positives, and false negatives.

    Args:
        expected_items: Ground truth rows.
        predicted_items: Predicted rows (may be shorter than expected).

    Returns:
        Tuple of (field_counts, mismatches, failed_rows) where field_counts
        maps field name to FieldCounts, mismatches contains a diagnostic for
        every false negative, and failed_rows maps field name to the list of
        row indices where that field had a false negative.
    """
    field_counts: dict[str, FieldCounts] = defaultdict(FieldCounts)
    mismatches: list[Mismatch] = []
    failed_rows: dict[str, list[int]] = defaultdict(list)

    for row_num, expected_row in enumerate(expected_items):
        predicted_row = predicted_items[row_num] if row_num < len(predicted_items) else {}

        expected_pairs = {(k, normalise_value(k, value=v)) for k, v in expected_row.items()}
        predicted_pairs = {(k, normalise_value(k, value=v)) for k, v in predicted_row.items()}

        matched = expected_pairs & predicted_pairs
        false_negatives = expected_pairs - predicted_pairs
        false_positives = predicted_pairs - expected_pairs

        for key, _ in matched:
            field_counts[key].tp += 1

        for key, _ in false_positives:
            field_counts[key].fp += 1

        predicted_by_key = dict(predicted_pairs)

        for key, expected_val in false_negatives:
            field_counts[key].fn += 1
            failed_rows[key].append(row_num)

            if key in predicted_by_key:
                predicted_val = predicted_by_key[key]
                dist = Levenshtein.distance(expected_val, predicted_val)
                pct_diff = (
                    _money_pct_diff(expected_val, predicted_val) if key in MONEY_FIELDS else None
                )
                mismatches.append(
                    Mismatch(
                        row=row_num,
                        key=key,
                        expected=expected_val,
                        predicted=predicted_val,
                        distance=dist,
                        pct_diff=pct_diff,
                    )
                )
            else:
                mismatches.append(
                    Mismatch(
                        row=row_num,
                        key=key,
                        expected=expected_val,
                        predicted="",
                        distance=-1,
                        missing=True,
                    )
                )

    # Extra predicted rows beyond expected count as false positives
    for row_num in range(len(expected_items), len(predicted_items)):
        predicted_row = predicted_items[row_num]
        for key in predicted_row:
            field_counts[key].fp += 1

    return dict(field_counts), mismatches, dict(failed_rows)


def calculate_micro_f1(
    field_counts: dict[str, FieldCounts],
) -> Metrics:
    """Compute micro-averaged precision, recall, and F1 from field counts.

    Args:
        field_counts: Mapping of field name to FieldCounts.

    Returns:
        Metrics with precision, recall, and f1.
    """
    total_tp = sum(c.tp for c in field_counts.values())
    total_fp = sum(c.fp for c in field_counts.values())
    total_fn = sum(c.fn for c in field_counts.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return Metrics(precision=precision, recall=recall, f1=f1)


def _row_label(row: int, *, row_labels: list[str] | None, short: bool = False) -> str:
    """Return a display label for a row index.

    Args:
        row: Zero-based row index.
        row_labels: Optional list of document names.
        short: If True, truncate hash-style filenames to first 8 characters.

    Returns:
        Document name if available, otherwise the row index as a string.
    """
    if row_labels is not None and row < len(row_labels):
        label = row_labels[row]
        if short:
            stem = Path(label).stem
            suffix = Path(label).suffix
            if len(stem) > 12:
                return stem[:8] + suffix
        return label
    return str(row)


def format_report(
    field_counts: dict[str, FieldCounts],
    mismatches: list[Mismatch],
    *,
    num_documents: int = 0,
    row_labels: list[str] | None = None,
) -> str:
    """Format scoring results as a human-readable report.

    Shows per-field summary with inline failure details (expected vs predicted)
    so every failure is explained in one place.

    Args:
        field_counts: Mapping of field name to FieldCounts.
        mismatches: List of Mismatch diagnostics (all failures).
        num_documents: Total number of documents scored.
        row_labels: Optional document names to label rows (e.g. PDF filenames).

    Returns:
        Multi-line report string.
    """
    lines: list[str] = []
    metrics = calculate_micro_f1(field_counts)

    total_tp = sum(c.tp for c in field_counts.values())
    total_fields = sum(c.tp + c.fn for c in field_counts.values())

    # Overall summary section
    overall_rows: list[tuple[str, str]] = []
    if num_documents > 0:
        overall_rows.append(("Documents", str(num_documents)))
    overall_rows.append(("F1", f"{metrics.f1:.3f}"))
    overall_rows.append(("Precision", f"{metrics.precision:.3f}"))
    overall_rows.append(("Recall", f"{metrics.recall:.3f}"))
    overall_rows.append(("Matched", f"{total_tp}/{total_fields} fields"))

    lines.extend(utils.format_table(("Overall", ""), overall_rows))

    # Index mismatches by field for inline display
    mismatches_by_field: dict[str, list[Mismatch]] = defaultdict(list)
    for m in mismatches:
        mismatches_by_field[m.key].append(m)

    # Per-field breakdown with inline failure details
    lines.append("")
    field_keys = sorted(field_counts)
    table_rows: list[tuple[str, str, str]] = []
    for key in field_keys:
        m = _field_metrics(field_counts[key])
        tp = field_counts[key].tp
        total = tp + field_counts[key].fn
        table_rows.append((key, f"{tp}/{total}", f"{m.f1:.3f}"))

    headers = ("Field", "Score", "F1")
    lines.extend(utils.format_table(headers, table_rows))

    # Failure details table
    detail_rows: list[tuple[str, ...]] = []
    for key in field_keys:
        for mm in mismatches_by_field.get(key, []):
            label = _row_label(mm.row, row_labels=row_labels, short=True)
            predicted = "-" if mm.missing else mm.predicted
            dist = "-" if mm.missing else str(mm.distance)
            if mm.pct_diff is not None:
                dist += f" ({mm.pct_diff})"
            detail_rows.append((key, label, mm.expected, predicted, dist))

    if detail_rows:
        lines.append("")
        detail_headers = ("Field", "Doc", "Expected", "Predicted", "d")
        lines.extend(utils.format_table(detail_headers, detail_rows))

    return "\n".join(lines)


def build_scores(
    field_counts: dict[str, FieldCounts],
) -> ScoresResult:
    """Build scoring results from field counts.

    Args:
        field_counts: Mapping of field name to FieldCounts.

    Returns:
        ScoresResult with overall and per-field metrics.
    """
    metrics = calculate_micro_f1(field_counts)
    total_tp = sum(c.tp for c in field_counts.values())
    total_fields = sum(c.tp + c.fn for c in field_counts.values())

    per_field: dict[str, PerFieldScore] = {}
    for key in sorted(field_counts):
        m = _field_metrics(field_counts[key])
        tp = field_counts[key].tp
        total = tp + field_counts[key].fn
        per_field[key] = PerFieldScore(tp=tp, total=total, f1=round(m.f1, 3))

    return ScoresResult(
        f1=round(metrics.f1, 3),
        precision=round(metrics.precision, 3),
        recall=round(metrics.recall, 3),
        matched=total_tp,
        total_fields=total_fields,
        per_field=per_field,
    )


def save_scores(scores: ScoresResult, *, path: Path | str) -> None:
    """Write scores to a JSON file.

    Args:
        scores: ScoresResult from build_scores.
        path: Destination file path.
    """
    with Path(path).open("w") as f:
        json.dump(dataclasses.asdict(scores), f, indent=2)
        f.write("\n")


def _read_row_labels(input_tsv: str) -> list[str]:
    """Read document names (first column) from an input TSV.

    Args:
        input_tsv: Path to the input TSV file.

    Returns:
        List of document name strings, one per row.
    """
    with Path(input_tsv).open() as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        return [row[0] if row else "?" for row in reader]


def run(*, expected: str, predicted: str, input_tsv: str | None = None) -> None:
    """Score predicted output against expected ground truth.

    Args:
        expected: Path to expected Kleister-format TSV.
        predicted: Path to predicted Kleister-format TSV.
        input_tsv: Optional path to input TSV (first column used as row labels).
    """
    expected_items = get_all_items(expected)
    predicted_items = get_all_items(predicted)
    row_labels = _read_row_labels(input_tsv) if input_tsv else None

    field_counts, mismatches, _failed_rows = compute_f1(
        expected_items, predicted_items=predicted_items
    )
    report = format_report(
        field_counts,
        mismatches,
        num_documents=len(expected_items),
        row_labels=row_labels,
    )
    print(report)

    # Save scores.json alongside the files if they live in an expts folder
    scores = build_scores(field_counts)
    for file_path in (expected, predicted):
        if "/expts/" in file_path or file_path.startswith("expts/"):
            scores_path = Path(file_path).parent / "scores.json"
            save_scores(scores, path=scores_path)
            logger.debug("Scores saved to %s", scores_path)
            break
