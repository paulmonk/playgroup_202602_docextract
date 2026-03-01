"""Tests for the experiment comparison module."""

from __future__ import annotations

import json

import pytest

from commands.compare import Experiment, _format_cells, _sort_key, format_table, run


def test_from_path_with_config_and_scores(tmp_path):
    expt = tmp_path / "20260101T00_00_00"
    expt.mkdir()
    (expt / "config.json").write_text(
        json.dumps(
            {
                "model": "anthropic/claude-3.5-haiku",
                "source": "combined",
                "num_documents": 5,
            }
        )
    )
    (expt / "scores.json").write_text(
        json.dumps({"f1": 0.85, "precision": 0.9, "recall": 0.8, "matched": 40, "total_fields": 50})
    )
    exp = Experiment.from_path(expt)
    assert exp is not None
    assert exp.model == "anthropic/claude-3.5-haiku"
    assert exp.source == "combined"
    assert exp.f1 == 0.85
    assert exp.matched == 40
    assert exp.total_fields == 50


def test_from_path_config_only_no_scores(tmp_path):
    expt = tmp_path / "20260101T00_00_00"
    expt.mkdir()
    (expt / "config.json").write_text(
        json.dumps({"model": "openai/gpt-4", "source": "pdf", "num_documents": 3})
    )
    exp = Experiment.from_path(expt)
    assert exp is not None
    assert exp.f1 is None
    assert exp.matched is None
    assert exp.model == "openai/gpt-4"


def test_from_path_no_config_returns_none(tmp_path):
    expt = tmp_path / "empty_folder"
    expt.mkdir()
    assert Experiment.from_path(expt) is None


def test_from_path_missing_config_fields_default(tmp_path):
    expt = tmp_path / "20260101T00_00_00"
    expt.mkdir()
    (expt / "config.json").write_text("{}")
    exp = Experiment.from_path(expt)
    assert exp is not None
    assert exp.model == "?"
    assert exp.source == "?"
    assert exp.num_documents == "?"


def test_from_path_corrupt_config_json_returns_none(tmp_path):
    """Corrupt config.json should return None, not crash."""
    expt = tmp_path / "20260101T00_00_00"
    expt.mkdir()
    (expt / "config.json").write_text("{invalid json")
    assert Experiment.from_path(expt) is None


def test_from_path_corrupt_scores_json_treats_as_unscored(tmp_path):
    """Corrupt scores.json should be treated as unscored, not crash."""
    expt = tmp_path / "20260101T00_00_00"
    expt.mkdir()
    (expt / "config.json").write_text(
        json.dumps({"model": "test/model", "source": "combined", "num_documents": 1})
    )
    (expt / "scores.json").write_text("{truncated")
    exp = Experiment.from_path(expt)
    assert exp is not None
    assert exp.f1 is None
    assert exp.matched is None


def test_sort_key_scored_above_unscored():
    scored = Experiment(folder="a", model="m", source="s", num_documents=1, f1=0.5)
    unscored = Experiment(folder="b", model="m", source="s", num_documents=1, f1=None)
    assert _sort_key(scored) > _sort_key(unscored)


def test_sort_key_higher_f1_first():
    high = Experiment(folder="a", model="m", source="s", num_documents=1, f1=0.9)
    low = Experiment(folder="b", model="m", source="s", num_documents=1, f1=0.3)
    assert _sort_key(high) > _sort_key(low)


def test_sort_key_equal_f1():
    a = Experiment(folder="a", model="m", source="s", num_documents=1, f1=0.5)
    b = Experiment(folder="b", model="m", source="s", num_documents=1, f1=0.5)
    assert _sort_key(a) == _sort_key(b)


def test_format_cells_with_scores():
    exp = Experiment(
        folder="20260101",
        model="anthropic/claude-3.5-haiku",
        source="combined",
        num_documents=10,
        f1=0.85,
        precision=0.9,
        recall=0.8,
        matched=40,
        total_fields=50,
    )
    cells = _format_cells(exp)
    assert cells[0] == "20260101"
    assert cells[4] == "0.850"
    assert cells[5] == "0.900"
    assert "40/50" in cells[7]
    assert "80%" in cells[7]


def test_format_cells_without_scores():
    exp = Experiment(folder="20260101", model="m", source="s", num_documents=5)
    cells = _format_cells(exp)
    assert cells[4] == "-"
    assert cells[5] == "-"
    assert cells[6] == "-"
    assert cells[7] == "-"


def test_format_cells_matched_without_total_fields_shows_dash():
    """When matched is set but total_fields is None, show dash not 'matched/None'."""
    exp = Experiment(
        folder="20260101",
        model="m",
        source="s",
        num_documents=5,
        f1=0.5,
        matched=5,
        total_fields=None,
    )
    cells = _format_cells(exp)
    assert cells[7] == "-"
    assert "None" not in cells[7]


def test_format_table_sorts_by_f1_descending():
    experiments = [
        Experiment(folder="low", model="m", source="s", num_documents=1, f1=0.3),
        Experiment(folder="high", model="m", source="s", num_documents=1, f1=0.9),
        Experiment(folder="mid", model="m", source="s", num_documents=1, f1=0.6),
    ]
    table = format_table(experiments)
    lines = table.splitlines()
    data_lines = lines[2:]
    assert data_lines[0].startswith("high")
    assert data_lines[1].startswith("mid")
    assert data_lines[2].startswith("low")


def test_format_table_header_present():
    experiments = [Experiment(folder="a", model="m", source="s", num_documents=1, f1=0.5)]
    table = format_table(experiments)
    header = table.splitlines()[0]
    assert "Timestamp" in header
    assert "Model" in header
    assert "F1" in header


def test_run_missing_dir_exits(tmp_path):
    with pytest.raises(SystemExit):
        run(expts_dir=str(tmp_path / "nonexistent"))


def test_run_no_experiments_exits(tmp_path):
    (tmp_path / "random_file.txt").write_text("not an experiment")
    with pytest.raises(SystemExit):
        run(expts_dir=str(tmp_path))


def test_run_prints_table(tmp_path, capsys):
    expt = tmp_path / "20260101T00_00_00"
    expt.mkdir()
    (expt / "config.json").write_text(
        json.dumps(
            {
                "model": "anthropic/claude-3.5-haiku",
                "source": "combined",
                "num_documents": 5,
            }
        )
    )
    (expt / "scores.json").write_text(json.dumps({"f1": 0.85, "precision": 0.9, "recall": 0.8}))
    run(expts_dir=str(tmp_path))
    output = capsys.readouterr().out
    assert "0.850" in output
    assert "anthropic/claude-3.5-haiku" in output
