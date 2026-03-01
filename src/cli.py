"""Extract structured fields from UK charity documents via LLM."""

from __future__ import annotations

import argparse
import logging
import sys

from commands import compare, pipeline, score


def _setup_logging(*, verbosity: int) -> None:
    """Configure root logger with verbosity-based level.

    Args:
        verbosity: 0 = WARNING, 1 = INFO, 2+ = DEBUG.
    """
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for info, -vv for debug).",
    )
    subparsers = parser.add_subparsers(dest="command")
    extract_parser = subparsers.add_parser("extract", help="Run the extraction pipeline.")
    extract_parser.add_argument(
        "--source",
        choices=sorted([*pipeline.SOURCE_COLUMN_BY_NAME, "pdf", "pdf-vision"]),
        default="combined",
        help=(
            "OCR column, 'pdf' for text extraction, or 'pdf-vision'"
            " for text+images (default: %(default)s)"
        ),
    )
    extract_parser.add_argument(
        "--model",
        default="anthropic/claude-3.5-haiku",
        help="OpenRouter model name (default: %(default)s)",
    )
    extract_parser.add_argument(
        "--input",
        dest="input_path",
        default="data/playgroup_dev_in.tsv",
        help="Input TSV path (default: %(default)s)",
    )
    extract_parser.add_argument(
        "--expected",
        default="data/playgroup_dev_expected_corrected.tsv",
        help="Expected TSV path for scoring (default: %(default)s)",
    )
    extract_parser.add_argument(
        "--doc",
        default=None,
        help="Filter to a single document by PDF filename substring (e.g. 'd07c46')",
    )
    score_parser = subparsers.add_parser("score", help="Score predicted output against expected.")
    score_parser.add_argument(
        "--expected",
        required=True,
        help="Path to expected Kleister-format TSV.",
    )
    score_parser.add_argument(
        "--predicted",
        required=True,
        help="Path to predicted Kleister-format TSV.",
    )
    score_parser.add_argument(
        "--input",
        dest="input_path",
        default=None,
        help="Path to input TSV (first column used as document labels in report).",
    )
    compare_parser = subparsers.add_parser("compare", help="Compare all experiment results.")
    compare_parser.add_argument(
        "--expts-dir",
        default="expts",
        help="Path to experiments directory (default: %(default)s)",
    )

    return parser


def main() -> None:
    """Parse args and dispatch to the appropriate subcommand."""
    parser = _build_parser()
    args = parser.parse_args()

    _setup_logging(verbosity=args.verbose)

    if args.command is None:
        parser.print_help()
        raise SystemExit(1)

    if args.command == "extract":
        pipeline.run(
            source=args.source,
            model=args.model,
            input_path=args.input_path,
            expected_path=args.expected,
            doc=args.doc,
        )
    elif args.command == "score":
        score.run(expected=args.expected, predicted=args.predicted, input_tsv=args.input_path)
    elif args.command == "compare":
        compare.run(expts_dir=args.expts_dir)


if __name__ == "__main__":
    main()
