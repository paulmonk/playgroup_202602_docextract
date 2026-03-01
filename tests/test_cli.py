"""Tests for the CLI argument parsing and dispatch."""

from __future__ import annotations

import pytest

from cli import _build_parser, main


def test_no_command_exits():
    with pytest.raises(SystemExit):
        main()


def test_extract_defaults():
    parser = _build_parser()
    args = parser.parse_args(["extract"])
    assert args.command == "extract"
    assert args.source == "combined"
    assert args.model == "anthropic/claude-3.5-haiku"
    assert args.input_path == "data/playgroup_dev_in.tsv"
    assert args.expected == "data/playgroup_dev_expected_corrected.tsv"
    assert args.doc is None


def test_extract_custom_args():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "extract",
            "--source",
            "pdf",
            "--model",
            "openai/gpt-4o",
            "--input",
            "custom_in.tsv",
            "--expected",
            "custom_exp.tsv",
            "--doc",
            "d07c46",
        ]
    )
    assert args.source == "pdf"
    assert args.model == "openai/gpt-4o"
    assert args.input_path == "custom_in.tsv"
    assert args.expected == "custom_exp.tsv"
    assert args.doc == "d07c46"


def test_extract_invalid_source_exits():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["extract", "--source", "nonexistent"])


def test_score_requires_expected_and_predicted():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["score"])


def test_score_args():
    parser = _build_parser()
    args = parser.parse_args(["score", "--expected", "e.tsv", "--predicted", "p.tsv"])
    assert args.command == "score"
    assert args.expected == "e.tsv"
    assert args.predicted == "p.tsv"
    assert args.input_path is None


def test_score_with_input():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "score",
            "--expected",
            "e.tsv",
            "--predicted",
            "p.tsv",
            "--input",
            "in.tsv",
        ]
    )
    assert args.input_path == "in.tsv"


def test_compare_defaults():
    parser = _build_parser()
    args = parser.parse_args(["compare"])
    assert args.command == "compare"
    assert args.expts_dir == "expts"


def test_compare_custom_dir():
    parser = _build_parser()
    args = parser.parse_args(["compare", "--expts-dir", "my_expts"])
    assert args.expts_dir == "my_expts"


def test_dispatch_extract(mocker):
    mocker.patch("sys.argv", ["cli", "extract"])
    mock_run = mocker.patch("cli.extract.run")
    main()
    mock_run.assert_called_once_with(
        source="combined",
        model="anthropic/claude-3.5-haiku",
        input_path="data/playgroup_dev_in.tsv",
        expected_path="data/playgroup_dev_expected_corrected.tsv",
        doc=None,
    )


def test_dispatch_score(mocker):
    mocker.patch("sys.argv", ["cli", "score", "--expected", "e.tsv", "--predicted", "p.tsv"])
    mock_run = mocker.patch("cli.score.run")
    main()
    mock_run.assert_called_once_with(expected="e.tsv", predicted="p.tsv", input_tsv=None)


def test_dispatch_compare(mocker):
    mocker.patch("sys.argv", ["cli", "compare"])
    mock_run = mocker.patch("cli.compare.run")
    main()
    mock_run.assert_called_once_with(expts_dir="expts")


def test_verbose_flag():
    parser = _build_parser()
    args = parser.parse_args(["-vv", "compare"])
    assert args.verbose == 2
