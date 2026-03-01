"""Three-step extraction pipeline for UK charity documents.

Steps: select_text (pick OCR source) -> extract (LLM call) -> validate (format checks).
Produces Kleister-format output (space-delimited key=value pairs per row).
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import logging
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, field_validator

from const import DATE_FIELDS, MONEY_FIELDS, UPPERCASE_FIELDS
from lib import llm_openrouter, pdf_extract, utils

# OCR text fields in the input TSV can be very large
csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)

# Map OCR source name to column index in the tab-delimited input TSV
SOURCE_COLUMN_BY_NAME: Final[Mapping[str, int]] = {
    "djvu2hocr": 2,
    "tesseract411": 3,
    "tesseract_march2020": 4,
    "combined": 5,
}

# UK postcode: 1-2 letters, 1-2 digits, optional digit/letter, space, digit, 2 letters
_UK_POSTCODE_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Z]{1,2}[0-9][0-9A-Z]?\s*[0-9][A-Z]{2}$",
    re.IGNORECASE,
)

# Matches content between triple backticks, with optional language tag
_BACKTICK_RE: Final[re.Pattern[str]] = re.compile(r"```(\w*)\n?(.*?)```", re.DOTALL)

# LLM system prompt, passed via the instructions/system role
SYSTEM_PROMPT: Final[str] = """\
You are an expert at extracting structured data from financial documents.
When a document contains figures at multiple levels of detail (subtotals
and grand totals), always select the grand total, not any subcategory.
Return only a JSON object with no prose and no markdown code fences."""

# LLM user prompt template; document text is appended in <document> tags
USER_PROMPT_TEMPLATE: Final[str] = """\
<task>
Extract the following fields from the document below. Return ONLY a JSON
object with field names as keys. Omit any field you cannot find.
</task>

<field_definitions>
- charity_number:
    Registered number. Integer, no leading zeros.
- report_date:
    End date of the reporting period (not the signing date). Format: YYYY-MM-DD.
- income_annually_in_british_pounds:
    Total annual income. Use the grand total line,
    not any subcategory such as voluntary income or investment income.
    If the statement units are in thousands (e.g. £000, £'000, or "in thousands"),
    multiply the extracted number by 1,000 before returning it.
    Format: number with 2 decimal places, no commas.
- spending_annually_in_british_pounds:
    Total annual expenditure. Use the grand total
    line, not any subcategory such as charitable activities, support costs, or governance
    costs. If multiple spending figures appear, sum them or find the row that already
    represents the overall total. If the statement units are in thousands (e.g. £000,
    £'000, or "in thousands"), multiply the extracted number by 1,000 before returning it.
    Format: number with 2 decimal places, no commas.
- address__post_town:
    Town from the registered/principal address, as written.
- address__postcode:
    Postcode from the registered/principal address, as written.
- address__street_line:
    Street name and number only from the registered/principal address.
    Do not include building names, floor numbers, unit numbers, or locality names.
    Example: "Ibex House, 42-47 Minories" -> "42-47 Minories" (exclude building name).
- charity_name:
    Full registered name as it appears in the document, including
    apostrophes.
</field_definitions>

<examples>
Example 1 (note: spending is the overall total, not the largest subcategory):
```json
{{
  "charity_number": "204567",
  "report_date": "2018-03-31",
  "income_annually_in_british_pounds": "89432.00",
  "spending_annually_in_british_pounds": "91205.00",
  "address__post_town": "BRISTOL",
  "address__postcode": "BS1 4DJ",
  "charity_name": "Community Aid Network"
}}
```

Example 2:
```json
{{
  "charity_number": "300703",
  "report_date": "2016-03-31",
  "income_annually_in_british_pounds": "41721.00",
  "spending_annually_in_british_pounds": "38246.00",
  "charity_name": "1st Bourne End Scout Group"
}}
```

</examples>
"""


class CharityExtraction(BaseModel):
    """Pydantic model for the 8 Kleister charity fields.

    All fields are optional since the LLM may not find every field.
    Validators check format and set invalid values to None.
    Unknown fields from the LLM response are silently dropped.
    """

    model_config = ConfigDict(extra="ignore")

    address__post_town: str | None = None
    address__postcode: str | None = None
    address__street_line: str | None = None
    charity_name: str | None = None
    charity_number: str | None = None
    income_annually_in_british_pounds: str | None = None
    report_date: str | None = None
    spending_annually_in_british_pounds: str | None = None

    @field_validator("charity_number", mode="before")
    @classmethod
    def check_charity_number(cls, v: object) -> str | None:
        """Validate charity number is 5-7 digits."""
        if v is None:
            return None
        digits = str(v).strip()
        if not digits.isdigit() or not (5 <= len(digits) <= 7):
            logger.debug("Rejected charity_number value: %r", v)
            return None
        return digits

    @field_validator("report_date", mode="before")
    @classmethod
    def check_report_date(cls, v: object) -> str | None:
        """Validate report date is ISO format and within 1990-2030."""
        if v is None:
            return None
        v_str = str(v).strip()
        try:
            parsed = dt.date.fromisoformat(v_str)
        except (ValueError, TypeError):
            logger.debug("Rejected report_date value (bad format): %r", v)
            return None
        if not (1990 <= parsed.year <= 2030):
            logger.debug("Rejected report_date value (out of range): %r", v)
            return None
        return v_str

    @field_validator("address__postcode", mode="before")
    @classmethod
    def check_postcode(cls, v: object) -> str | None:
        """Validate UK postcode format."""
        if v is None:
            return None
        v_str = str(v).strip()
        raw_for_check = v_str.upper().replace("_", " ")
        if not _UK_POSTCODE_RE.match(raw_for_check):
            logger.debug("Rejected postcode value: %r", v)
            return None
        return v_str

    @field_validator(
        "income_annually_in_british_pounds",
        "spending_annually_in_british_pounds",
        mode="before",
    )
    @classmethod
    def check_money(cls, v: object) -> str | None:
        """Validate monetary value is a positive number."""
        if v is None:
            return None
        v_str = str(v).strip()
        try:
            amount = float(v_str.replace(",", ""))
        except (ValueError, TypeError):
            logger.debug("Rejected money value (not numeric): %r", v)
            return None
        if amount <= 0:
            logger.debug("Rejected money value (not positive): %r", v)
            return None
        return v_str

    @field_validator("address__post_town", "address__street_line", "charity_name", mode="before")
    @classmethod
    def stringify(cls, v: object) -> str | None:
        """Convert non-None values to strings."""
        if v is None:
            return None
        return str(v)

    def to_fields(self) -> dict[str, str]:
        """Return non-None fields as a dict."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


@dataclasses.dataclass(kw_only=True, frozen=True)
class ExperimentConfig:
    """Metadata for a single experiment run."""

    model: str
    source: str
    input_file: str
    num_documents: int
    timestamp: str
    git_commit: str
    expected_file: str


def _get_git_commit() -> str:
    """Return the short git commit hash, or ``"unknown"`` on failure."""
    try:
        result = subprocess.run(
            ["/usr/bin/git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _extract_from_triple_backticks(text: str) -> str | None:
    """Return the content from the last triple-backtick block, or None.

    Args:
        text: Raw text that may contain triple-backtick fenced blocks.

    Returns:
        The extracted content string, or None if no backtick block found.
    """
    matches = _BACKTICK_RE.findall(text)
    if not matches:
        return None

    lang_or_content, content = matches[-1]

    # If content is empty, the first group was the content (not a language tag)
    if not content.strip() and lang_or_content:  # noqa: SIM108 - ternary would be unreadable here
        result = lang_or_content.strip()
    else:
        result = content.strip()

    # Strip leading comment markers from some model outputs
    if result.startswith("///"):
        result = result[3:].strip()
    elif result.startswith("//"):
        result = result[2:].strip()

    return result


def _parse_json_object_from_mixed_text(text: str) -> dict | None:
    """Parse the first JSON object found in free-form text.

    Args:
        text: Text that may contain prose plus an inline JSON object.

    Returns:
        Parsed dict if found, else None.
    """
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        # Skip JSON arrays ([ followed by {, possibly with whitespace between)
        prev = idx - 1
        while prev >= 0 and text[prev] in " \t\n\r":
            prev -= 1
        if prev >= 0 and text[prev] == "[":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_json_dict(raw: str) -> dict | None:
    """Try multiple strategies to get a JSON dict from LLM output.

    Args:
        raw: Raw LLM response text.

    Returns:
        Parsed dict if found, else None.
    """
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    extracted = _extract_from_triple_backticks(raw)
    if extracted:
        try:
            parsed = json.loads(extracted)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    return _parse_json_object_from_mixed_text(raw)


def select_text(row: list[str] | tuple[str, ...], *, source: str = "combined") -> str:
    """Pick the OCR text column for the given source name.

    Args:
        row: A single row from the input TSV.
        source: OCR source name (e.g. ``combined``, ``djvu2hocr``).

    Returns:
        The OCR text for the requested source.

    Raises:
        ValueError: If the source is unknown or the row is too short.
    """
    col_index = SOURCE_COLUMN_BY_NAME.get(source)
    if col_index is None:
        raise ValueError(f"Unknown source {source!r}; choose from {sorted(SOURCE_COLUMN_BY_NAME)}")
    if col_index >= len(row):
        raise ValueError(
            f"Row has {len(row)} columns, need at least {col_index + 1} for source {source!r}"
        )
    return row[col_index]


def extract(
    text: str,
    *,
    model: str,
    images: list[str] | None = None,
) -> dict[str, str]:
    """Call the LLM to extract structured fields from OCR text.

    Args:
        text: The OCR document text.
        model: OpenRouter model name.
        images: Optional base64-encoded PNG page images for vision models.

    Returns:
        Dict of extracted field name to value, or empty dict on failure.
    """
    user_prompt = f"{USER_PROMPT_TEMPLATE}<document>{text}\n</document>"
    raw = llm_openrouter.call_llm(
        model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        images=images,
    )

    if not raw:
        logger.warning("LLM returned empty response")
        return {}

    return parse_llm_response(raw)


def parse_llm_response(raw: str) -> dict[str, str]:
    """Parse and validate JSON from LLM response.

    Tries direct parse, triple-backtick extraction, then embedded JSON object.
    Uses the CharityExtraction model to filter unknown fields and validate
    field formats in one step.

    Args:
        raw: Raw LLM response text.

    Returns:
        Dict of recognised, valid field name to string value, or empty dict.
    """
    parsed = _extract_json_dict(raw)
    if parsed is None:
        logger.warning("Failed to parse JSON from LLM response: %.200s", raw)
        return {}
    fields = CharityExtraction.model_validate(parsed).to_fields()
    if parsed and not fields:
        logger.warning(
            "All %d extracted fields failed validation: %s",
            len(parsed),
            sorted(parsed.keys()),
        )
    return fields


def normalise_for_output(key: str, *, value: str) -> str:
    """Normalise a single field value for Kleister output format.

    Args:
        key: The field name, used to dispatch normalisation rules.
        value: The raw field value from the LLM.

    Returns:
        Value formatted for Kleister output (underscores, casing, etc.).
    """
    # Newlines inside a value would break TSV output
    value = value.replace("\n", " ").replace("\r", " ")
    # Normalise smart/curly quotes to straight apostrophes and quotes
    value = value.replace("\u2018", "'").replace("\u2019", "'")
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    if key in MONEY_FIELDS:
        return utils.normalise_money(value)
    if key in DATE_FIELDS:
        return utils.normalise_date(value)
    if key == "charity_number":
        try:
            return str(int(value))
        except ValueError:
            return value
    if key in UPPERCASE_FIELDS:
        return value.upper().replace(" ", "_")
    return value.replace(" ", "_")


def format_row(fields: dict[str, str]) -> str:
    """Format a dict of fields as a Kleister output line.

    Args:
        fields: Dict of field name to value.

    Returns:
        Tab-delimited key=value string, sorted alphabetically by key.
    """
    return "\t".join(f"{k}={normalise_for_output(k, value=fields[k])}" for k in sorted(fields))


def write_output(rows: list[str], *, output_path: str) -> None:
    """Write formatted rows to the output TSV file.

    Args:
        rows: List of Kleister-format output lines.
        output_path: Destination file path.
    """
    with Path(output_path).open("w") as f:
        f.writelines(row + "\n" for row in rows)


def _create_timestamped_folder() -> str:
    """Create an experiment folder named with current UTC datetime.

    Returns:
        The folder name (not full path), e.g. ``20260226T15_30_00``.
    """
    folder_name = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H_%M_%S")
    (Path("expts") / folder_name).mkdir(parents=True)
    return folder_name


def save_config(
    *,
    folder: str,
    model: str,
    source: str,
    input_path: str,
    num_documents: int,
    git_commit: str,
    expected_file: str,
) -> None:
    """Save experiment metadata as config.json.

    Args:
        folder: Experiment folder name inside expts/.
        model: OpenRouter model name.
        source: OCR source name.
        input_path: Path to the input TSV.
        num_documents: Number of documents processed.
        git_commit: Short git commit hash.
        expected_file: Path to the expected TSV.
    """
    config = ExperimentConfig(
        model=model,
        source=source,
        input_file=input_path,
        num_documents=num_documents,
        timestamp=dt.datetime.now(dt.UTC).isoformat(),
        git_commit=git_commit,
        expected_file=expected_file,
    )
    config_path = Path("expts") / folder / "config.json"
    with config_path.open("w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)
        f.write("\n")


def _write_expected_subset(
    expected_path: str,
    *,
    folder: str,
    selected_indices: tuple[int, ...] | None,
) -> None:
    """Write expected rows matching the selected input rows to the experiment folder.

    Args:
        expected_path: Path to the full expected TSV file.
        folder: Experiment folder name under expts/.
        selected_indices: Row indices to include, or None for all rows.
    """
    with Path(expected_path).open() as f:
        all_lines = f.readlines()

    if selected_indices is not None:
        out_of_bounds = [i for i in selected_indices if i >= len(all_lines)]
        if out_of_bounds:
            logger.warning(
                "Expected file %s has %d rows but input references rows %s; "
                "these will be missing from the expected subset",
                expected_path,
                len(all_lines),
                out_of_bounds,
            )
        lines = [all_lines[i] for i in selected_indices if i < len(all_lines)]
    else:
        lines = all_lines

    dest = Path("expts") / folder / "expected.tsv"
    with dest.open("w") as f:
        f.writelines(lines)
    logger.debug("Wrote %d expected rows to %s", len(lines), dest)


def run(
    *,
    source: str,
    model: str,
    input_path: str,
    expected_path: str,
    doc: str | None = None,
) -> str:
    """Read input, run pipeline on each row, write output and metadata.

    Args:
        source: OCR source column name.
        model: OpenRouter model name.
        input_path: Path to the input TSV.
        expected_path: Path to the expected TSV (for writing a matching subset).
        doc: If set, only process rows whose PDF filename contains this substring.

    Returns:
        Path to the output TSV file.
    """
    with Path(input_path).open() as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        all_rows = list(reader)

    if doc:
        indexed_rows = [(i, r) for i, r in enumerate(all_rows) if r and doc in r[0]]
        if not indexed_rows:
            raise SystemExit(f"No documents matching {doc!r} found in {input_path}")
        selected_indices, rows = zip(*indexed_rows, strict=True)
        rows = list(rows)
    else:
        selected_indices = None
        rows = all_rows

    logger.info("Loaded %d documents from %s", len(rows), input_path)

    # Create experiment folder and attach per-experiment LLM log
    folder = _create_timestamped_folder()
    # Capture full prompts and responses at DEBUG level in the log file.
    # propagate=False stops them reaching the console handler.
    llm_logger = logging.getLogger("lib.llm_openrouter")
    prev_level = llm_logger.level
    llm_logger.setLevel(logging.DEBUG)
    llm_logger.propagate = False
    log_path = Path("expts") / folder / "llm_calls.log"
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    llm_logger.addHandler(fh)

    try:
        output_lines: list[str] = []
        for i, row in enumerate(rows):
            pdf_name = row[0] if row else f"row_{i}"
            logger.info("Processing %d/%d: %s", i + 1, len(rows), pdf_name)

            try:
                images = None
                if source in ("pdf", "pdf-vision"):
                    pdf_path = f"data/{pdf_name}"
                    text = pdf_extract.extract_pdf_text(pdf_path)
                    if source == "pdf-vision":
                        images = pdf_extract.render_pages_as_base64(pdf_path)
                else:
                    text = select_text(row, source=source)
                fields = extract(text, model=model, images=images)
            except Exception:
                logger.exception(
                    "Failed to process document %s (%d/%d)",
                    pdf_name,
                    i + 1,
                    len(rows),
                )
                fields = {}

            line = format_row(fields)
            output_lines.append(line)
            logger.info("Extracted %d fields for %s", len(fields), pdf_name)

        output_path = str(Path("expts") / folder / "extracted.tsv")
        write_output(output_lines, output_path=output_path)
        _write_expected_subset(expected_path, folder=folder, selected_indices=selected_indices)
        save_config(
            folder=folder,
            model=model,
            source=source,
            input_path=input_path,
            num_documents=len(rows),
            git_commit=_get_git_commit(),
            expected_file=expected_path,
        )
    finally:
        llm_logger.removeHandler(fh)
        llm_logger.setLevel(prev_level)
        llm_logger.propagate = True
        fh.close()

    logger.info("Results written to expts/%s/", folder)
    return output_path
