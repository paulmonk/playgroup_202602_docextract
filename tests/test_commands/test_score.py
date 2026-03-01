"""Tests for the field-type-aware F1 scorer."""

from __future__ import annotations

import pytest

from commands.score import (
    FieldCounts,
    _field_metrics,
    _money_pct_diff,
    build_scores,
    calculate_micro_f1,
    compute_f1,
    format_report,
    get_all_items,
    normalise_value,
    save_scores,
)
from lib.utils import MONEY_FIELDS


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        # Money fields: strip commas, format to 2dp
        ("income_annually_in_british_pounds", "255653.00", "255653.00"),
        ("income_annually_in_british_pounds", "255653", "255653.00"),
        ("income_annually_in_british_pounds", "1,234,567.89", "1234567.89"),
        ("spending_annually_in_british_pounds", "29113", "29113.00"),
        ("spending_annually_in_british_pounds", "0", "0.00"),
        # Date fields: parse to ISO format
        ("report_date", "2015-12-31", "2015-12-31"),
        ("report_date", "31/12/2015", "2015-12-31"),
        ("report_date", "31-12-2015", "2015-12-31"),
        # General fields: uppercase
        ("charity_name", "The_Sanata_Charitable_Trust", "THE_SANATA_CHARITABLE_TRUST"),
        ("charity_number", "1132766", "1132766"),
        ("address__postcode", "SY3_7PQ", "SY3_7PQ"),
        ("address__postcode", "sy3_7pq", "SY3_7PQ"),
    ],
)
def test_normalise_value(key, value, expected):
    assert normalise_value(key, value=value) == expected


def test_normalise_money_with_whitespace():
    assert normalise_value("income_annually_in_british_pounds", value=" 100.5 ") == "100.50"


def test_normalise_money_non_numeric_fallback():
    assert normalise_value("income_annually_in_british_pounds", value="N/A") == "N/A"


def test_normalise_date_unrecognised_format():
    assert normalise_value("report_date", value="Dec 2015") == "DEC 2015"


def test_perfect_match():
    expected = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
    predicted = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
    field_counts, near_misses, _ = compute_f1(expected, predicted_items=predicted)
    metrics = calculate_micro_f1(field_counts)
    assert metrics.f1 == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert near_misses == []


def test_partial_match():
    expected = [{"charity_number": "123", "charity_name": "FOO"}]
    predicted = [{"charity_number": "123", "charity_name": "BAR"}]
    field_counts, _near_misses, _ = compute_f1(expected, predicted_items=predicted)
    assert field_counts["charity_number"].tp == 1
    assert field_counts["charity_name"].fn == 1
    assert field_counts["charity_name"].fp == 1
    metrics = calculate_micro_f1(field_counts)
    assert metrics.f1 == pytest.approx(0.5)


def test_empty_predictions():
    expected = [{"a": "1", "b": "2"}]
    predicted: list[dict[str, str]] = []
    field_counts, *_ = compute_f1(expected, predicted_items=predicted)
    metrics = calculate_micro_f1(field_counts)
    assert metrics.f1 == 0.0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0


def test_extra_predictions():
    expected = [{"a": "1"}]
    predicted = [{"a": "1", "b": "2"}]
    field_counts, *_ = compute_f1(expected, predicted_items=predicted)
    metrics = calculate_micro_f1(field_counts)
    assert metrics.recall == 1.0
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(2 / 3)


def test_empty_both():
    field_counts, near_misses, _ = compute_f1([], predicted_items=[])
    metrics = calculate_micro_f1(field_counts)
    assert metrics.f1 == 0.0
    assert near_misses == []


def test_predicted_shorter_than_expected():
    expected = [{"a": "1"}, {"a": "2"}, {"a": "3"}]
    predicted = [{"a": "1"}]
    field_counts, *_ = compute_f1(expected, predicted_items=predicted)
    assert field_counts["a"].tp == 1
    assert field_counts["a"].fn == 2
    metrics = calculate_micro_f1(field_counts)
    assert metrics.recall == pytest.approx(1 / 3)
    assert metrics.precision == 1.0


def test_near_miss_detected():
    expected = [{"charity_number": "1064606"}]
    predicted = [{"charity_number": "064606"}]
    _, near_misses, _ = compute_f1(expected, predicted_items=predicted)
    assert len(near_misses) == 1
    miss = near_misses[0]
    assert miss.key == "charity_number"
    assert miss.distance == 1


def test_near_miss_money_pct_diff():
    expected = [{"income_annually_in_british_pounds": "100.00"}]
    predicted = [{"income_annually_in_british_pounds": "90.00"}]
    _, near_misses, _ = compute_f1(expected, predicted_items=predicted)
    assert len(near_misses) == 1
    assert near_misses[0].pct_diff == "10.0%"
    assert near_misses[0].key in MONEY_FIELDS


def test_missing_field_generates_mismatch():
    expected = [{"charity_number": "123"}]
    predicted = [{"charity_name": "FOO"}]
    _, mismatches, _ = compute_f1(expected, predicted_items=predicted)
    charity_num_mismatches = [m for m in mismatches if m.key == "charity_number"]
    assert len(charity_num_mismatches) == 1
    assert charity_num_mismatches[0].missing is True
    assert charity_num_mismatches[0].expected == "123"


def test_money_normalisation_in_f1():
    expected = [{"spending_annually_in_british_pounds": "291134.00"}]
    predicted = [{"spending_annually_in_british_pounds": "29113"}]
    field_counts, near_misses, _ = compute_f1(expected, predicted_items=predicted)
    assert field_counts["spending_annually_in_british_pounds"].fn == 1
    assert len(near_misses) == 1
    assert near_misses[0].pct_diff is not None


def test_money_format_equivalence():
    expected = [{"income_annually_in_british_pounds": "255653"}]
    predicted = [{"income_annually_in_british_pounds": "255653.00"}]
    field_counts, *_ = compute_f1(expected, predicted_items=predicted)
    metrics = calculate_micro_f1(field_counts)
    assert metrics.f1 == 1.0


def test_get_all_items(tmp_path):
    content = "charity_name=Foo\tcharity_number=123\ncharity_name=Bar\tcharity_number=456\n"
    filepath = tmp_path / "test.tsv"
    filepath.write_text(content)
    items = get_all_items(str(filepath))
    assert len(items) == 2
    assert items[0] == {"charity_name": "Foo", "charity_number": "123"}
    assert items[1] == {"charity_name": "Bar", "charity_number": "456"}


def test_get_all_items_missing_field(tmp_path):
    content = "a=1\tb=2\tc=3\na=4\tb=5\n"
    filepath = tmp_path / "test.tsv"
    filepath.write_text(content)
    items = get_all_items(str(filepath))
    assert len(items) == 2
    assert items[0] == {"a": "1", "b": "2", "c": "3"}
    assert items[1] == {"a": "4", "b": "5"}


def test_self_comparison_perfect_f1():
    """Scoring expected against itself must yield F1 of 1.0."""
    items = [
        {
            "charity_number": "1132766",
            "income_annually_in_british_pounds": "255653.00",
            "report_date": "2015-12-31",
            "charity_name": "The_Sanata_Charitable_Trust",
        },
        {
            "charity_number": "1039369",
            "income_annually_in_british_pounds": "122836.00",
            "report_date": "2013-12-31",
            "charity_name": "Cantate_Youth_Choir",
        },
    ]
    field_counts, near_misses, _ = compute_f1(items, predicted_items=items)
    metrics = calculate_micro_f1(field_counts)
    assert metrics.f1 == 1.0
    assert near_misses == []


def test_field_metrics_perfect():
    counts = FieldCounts(tp=10, fp=0, fn=0)
    m = _field_metrics(counts)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_field_metrics_no_matches():
    counts = FieldCounts(tp=0, fp=5, fn=5)
    m = _field_metrics(counts)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_field_metrics_partial():
    counts = FieldCounts(tp=5, fp=5, fn=0)
    m = _field_metrics(counts)
    assert m.precision == pytest.approx(0.5)
    assert m.recall == 1.0
    assert m.f1 == pytest.approx(2 / 3)


def test_field_metrics_empty():
    counts = FieldCounts(tp=0, fp=0, fn=0)
    m = _field_metrics(counts)
    assert m.f1 == 0.0


def test_money_pct_diff_normal():
    assert _money_pct_diff("100.00", "90.00") == "10.0%"


def test_money_pct_diff_zero_expected_nonzero_predicted():
    assert _money_pct_diff("0.0", "100.0") == "inf"


def test_money_pct_diff_both_zero():
    assert _money_pct_diff("0.0", "0.0") == "N/A"


def test_money_pct_diff_non_numeric():
    assert _money_pct_diff("abc", "100") == "N/A"


def test_format_report_contains_overall_f1():
    field_counts = {"a": FieldCounts(tp=5, fp=1, fn=2)}
    report = format_report(field_counts, [])
    assert "Overall" in report
    assert "0.714" in report  # F1 for tp=5, fp=1, fn=2
    assert "5/7" in report


def test_format_report_per_field_lines():
    field_counts = {
        "charity_number": FieldCounts(tp=3, fp=0, fn=1),
        "charity_name": FieldCounts(tp=4, fp=0, fn=0),
    }
    report = format_report(field_counts, [])
    assert "charity_name" in report
    assert "charity_number" in report
    assert "Field" in report


def test_format_report_mismatch_detail_shown():
    from commands.score import Mismatch

    field_counts = {"a": FieldCounts(tp=0, fp=0, fn=1)}
    mismatches = [Mismatch(row=0, key="a", expected="foo", predicted="fob", distance=1)]
    report = format_report(field_counts, mismatches)
    assert "Expected" in report
    assert "Predicted" in report
    assert "foo" in report
    assert "fob" in report


def test_format_report_missing_field_shown():
    from commands.score import Mismatch

    field_counts = {"a": FieldCounts(tp=0, fp=0, fn=1)}
    mismatches = [Mismatch(row=0, key="a", expected="foo", predicted="", distance=-1, missing=True)]
    report = format_report(field_counts, mismatches)
    # Missing fields show "-" in Predicted column
    lines = report.splitlines()
    detail_line = [ln for ln in lines if "foo" in ln and ln.strip().startswith("a")]
    assert len(detail_line) == 1
    assert "-" in detail_line[0]


def test_build_scores_perfect():
    field_counts = {"a": FieldCounts(tp=5, fp=0, fn=0)}
    scores = build_scores(field_counts)
    assert scores.f1 == 1.0
    assert scores.precision == 1.0
    assert scores.recall == 1.0
    assert scores.matched == 5
    assert scores.total_fields == 5
    assert "a" in scores.per_field
    assert scores.per_field["a"].f1 == 1.0


def test_build_scores_partial():
    field_counts = {"a": FieldCounts(tp=3, fp=1, fn=2)}
    scores = build_scores(field_counts)
    assert 0 < scores.f1 < 1
    assert scores.matched == 3
    assert scores.total_fields == 5


def test_save_scores_writes_json(tmp_path):
    import json

    field_counts = {"a": FieldCounts(tp=5, fp=0, fn=0)}
    scores = build_scores(field_counts)
    out = tmp_path / "scores.json"
    save_scores(scores, path=out)
    data = json.loads(out.read_text())
    assert data["f1"] == 1.0
    assert data["matched"] == 5
    assert "per_field" in data


def test_get_all_items_malformed_row_raises(tmp_path):
    """A row missing '=' should raise ValueError with context."""
    filepath = tmp_path / "bad.tsv"
    filepath.write_text("a=1\tb=2\na=3\tNOEQUALS\n")
    with pytest.raises(ValueError, match="Malformed row 1"):
        get_all_items(str(filepath))


def test_get_all_items_empty_cells_skipped(tmp_path):
    """Empty cells (e.g. trailing tabs) should not crash."""
    filepath = tmp_path / "trailing.tsv"
    filepath.write_text("a=1\t\n")
    items = get_all_items(str(filepath))
    assert items == [{"a": "1"}]


def test_extra_predicted_rows_count_as_fp():
    """Extra predicted rows beyond expected should penalise precision."""
    expected = [{"a": "1"}]
    predicted = [{"a": "1"}, {"a": "2", "b": "3"}]
    field_counts, *_ = compute_f1(expected, predicted_items=predicted)
    assert field_counts["a"].fp == 1
    assert field_counts["b"].fp == 1
    metrics = calculate_micro_f1(field_counts)
    assert metrics.recall == 1.0
    assert metrics.precision < 1.0


def test_run_auto_saves_scores_for_expts_path(tmp_path):
    """score.run() should write scores.json when paths contain expts/."""
    from commands.score import run

    expts = tmp_path / "expts" / "20260101T00_00_00"
    expts.mkdir(parents=True)
    content = "charity_name=Foo\tcharity_number=123\n"
    expected_file = expts / "expected.tsv"
    predicted_file = expts / "extracted.tsv"
    expected_file.write_text(content)
    predicted_file.write_text(content)

    run(expected=str(expected_file), predicted=str(predicted_file))

    scores_path = expts / "scores.json"
    assert scores_path.exists()
    import json

    data = json.loads(scores_path.read_text())
    assert data["f1"] == 1.0


def test_run_no_save_when_not_in_expts(tmp_path):
    """score.run() should not write scores.json outside expts/ paths."""
    content = "charity_name=Foo\n"
    expected_file = tmp_path / "expected.tsv"
    predicted_file = tmp_path / "predicted.tsv"
    expected_file.write_text(content)
    predicted_file.write_text(content)

    from commands.score import run

    run(expected=str(expected_file), predicted=str(predicted_file))

    assert not (tmp_path / "scores.json").exists()


def test_self_comparison_with_real_data(tmp_path):
    """Scoring the actual expected file against itself must yield F1 of 1.0."""
    import shutil

    src = "data/playgroup_dev_expected.tsv"
    dst = tmp_path / "copy.tsv"
    shutil.copy(src, dst)
    expected = get_all_items(src)
    predicted = get_all_items(str(dst))
    field_counts, near_misses, _ = compute_f1(expected, predicted_items=predicted)
    metrics = calculate_micro_f1(field_counts)
    assert metrics.f1 == 1.0
    assert near_misses == []
