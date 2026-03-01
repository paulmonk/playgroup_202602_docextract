"""Compare experiment results across models.

Reads config.json and scores.json from each experiment folder under expts/
and prints a sorted comparison table.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from lib import utils

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=True)
class Experiment:
    """A single experiment's config and scoring results."""

    folder: str
    model: str
    source: str
    num_documents: int | str
    f1: float | None = None
    precision: float | None = None
    recall: float | None = None
    matched: int | None = None
    total_fields: int | None = None

    @classmethod
    def from_path(cls, path: Path) -> Experiment | None:
        """Load config and scores from an experiment directory.

        Args:
            path: Full path to the experiment folder.

        Returns:
            Experiment instance, or None if config.json is missing.
        """
        config_path = path / "config.json"
        scores_path = path / "scores.json"

        try:
            with config_path.open() as f:
                config = json.load(f)
        except (FileNotFoundError, NotADirectoryError):
            return None
        except json.JSONDecodeError as exc:
            logger.warning("Corrupt config.json in %s: %s", path, exc)
            return None

        scores: dict[str, float | int] = {}
        try:
            with scores_path.open() as f:
                scores = json.load(f)
        except (FileNotFoundError, NotADirectoryError):
            pass
        except json.JSONDecodeError as exc:
            logger.warning("Corrupt scores.json in %s, treating as unscored: %s", path, exc)

        matched_raw = scores.get("matched")
        total_raw = scores.get("total_fields")

        return cls(
            folder=path.name,
            model=config.get("model", "?"),
            source=config.get("source", "?"),
            num_documents=config.get("num_documents", "?"),
            f1=scores.get("f1"),
            precision=scores.get("precision"),
            recall=scores.get("recall"),
            matched=int(matched_raw) if matched_raw is not None else None,
            total_fields=int(total_raw) if total_raw is not None else None,
        )


def _sort_key(exp: Experiment) -> tuple[bool, float]:
    """Sort key: scored experiments first, then by F1 descending."""
    return (exp.f1 is not None, exp.f1 or 0)


def _format_cells(exp: Experiment) -> tuple[str, ...]:
    """Format an experiment as a row of display strings.

    Args:
        exp: Experiment to format.

    Returns:
        Tuple of cell strings matching the table headers.
    """
    f1 = f"{exp.f1:.3f}" if exp.f1 is not None else "-"
    prec = f"{exp.precision:.3f}" if exp.precision is not None else "-"
    recall = f"{exp.recall:.3f}" if exp.recall is not None else "-"
    if exp.matched is not None and exp.total_fields is not None:
        pct = exp.matched / exp.total_fields * 100 if exp.total_fields else 0
        fields = f"{exp.matched}/{exp.total_fields} ({pct:.0f}%)"
    else:
        fields = "-"

    return (
        exp.folder,
        exp.model,
        exp.source,
        str(exp.num_documents),
        f1,
        prec,
        recall,
        fields,
    )


def format_table(experiments: list[Experiment]) -> str:
    """Format experiments as a comparison table sorted by F1 descending.

    Args:
        experiments: List of Experiment instances.

    Returns:
        Formatted table string.
    """
    experiments.sort(key=_sort_key, reverse=True)

    headers = ("Timestamp", "Model", "Source", "Docs", "F1", "Prec", "Recall", "Matched")
    rows = [_format_cells(exp) for exp in experiments]

    return "\n".join(utils.format_table(headers, rows))


def run(*, expts_dir: str = "expts") -> None:
    """Scan an experiments directory and print comparison table.

    Args:
        expts_dir: Path to the experiments directory.
    """
    root = Path(expts_dir)
    if not root.is_dir():
        logger.error("No %s/ directory found.", expts_dir)
        raise SystemExit(1)

    experiments = []
    for child in sorted(root.iterdir()):
        exp = Experiment.from_path(child)
        if exp is not None:
            experiments.append(exp)

    if not experiments:
        logger.error("No experiments found.")
        raise SystemExit(1)

    print(format_table(experiments))
