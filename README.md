# playgroup-docextract

Fork of [ianozsvald/playgroup_202602_docextract](https://github.com/ianozsvald/playgroup_202602_docextract).

Extract structured fields from UK charity financial PDFs using LLMs,
then score and compare results across models and text sources.

Built on a subset of the [Kleister Charity](https://github.com/applicaai/kleister-charity)
dataset: 11 documents ranging from short to 200+ pages, drawn from a
heterogeneous corpus of ~3,000 UK charity reports.

## Extracted fields

| Field | Format |
|---|---|
| `charity_number` | Integer, no leading zeros |
| `charity_name` | Full registered name |
| `report_date` | `YYYY-MM-DD` |
| `income_annually_in_british_pounds` | Number, 2 decimal places |
| `spending_annually_in_british_pounds` | Number, 2 decimal places |
| `address__post_town` | Town name |
| `address__postcode` | UK postcode |
| `address__street_line` | Street and number only |

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
echo 'OPENROUTER_API_KEY=your-key-here' > .env
```

## Usage

The CLI has three commands: `extract`, `score`, and `compare`.

### Run an extraction

```bash
just extract --model anthropic/claude-3.5-haiku --source pdf
```

Text sources: `combined` (default), `djvu2hocr`, `tesseract411`,
`tesseract_march2020`, `pdf` (PyMuPDF text extraction), `pdf-vision`
(text + images).

Each run creates a timestamped folder under `expts/` with the
extracted TSV, config, LLM call log, and scores.

### Score a single experiment

```bash
just score-expt 20260227T15_50_28
```

```
Overall
---------  ------------
Documents  11
F1         0.902
Precision  0.886
Recall     0.918
Matched    78/85 fields

Field                                Score  F1
-----------------------------------  -----  -----
address__post_town                   10/11  0.909
address__postcode                    10/10  0.952
address__street_line                 8/9    0.800
charity_name                         6/11   0.545
charity_number                       11/11  1.000
income_annually_in_british_pounds    11/11  1.000
report_date                          11/11  1.000
spending_annually_in_british_pounds  11/11  1.000
```

Failures are shown with expected vs predicted values and edit distance.

### Compare all experiments

```bash
just compare
```

```
Timestamp          Model                          Source      Docs  F1     Prec   Recall  Matched
-----------------  -----------------------------  ----------  ----  -----  -----  ------  -----------
20260227T15_50_28  google/gemini-3-flash-preview  pdf         11    0.902  0.886  0.918   78/85 (92%)
20260227T16_02_17  google/gemini-3-flash-preview  pdf-vision  11    0.902  0.886  0.918   78/85 (92%)
20260227T15_47_27  anthropic/claude-haiku-4.5     pdf         11    0.879  0.864  0.894   76/85 (89%)
20260227T15_31_50  anthropic/claude-3.5-haiku     pdf         11    0.872  0.862  0.882   75/85 (88%)
20260227T15_42_49  google/gemini-2.5-flash        pdf         11    0.860  0.851  0.871   74/85 (87%)
20260227T15_43_24  openai/gpt-4o-mini             pdf         11    0.763  0.750  0.776   66/85 (78%)
```

(Sorted by F1 descending, truncated for brevity.)

## Development

```bash
just check    # lint and format via pre-commit
just test     # pytest
```

## Data

In `data/`:

- `playgroup_dev_in.tsv` -- input rows (PDF filename + OCR text columns)
- `playgroup_dev_expected.tsv` -- ground truth in Kleister format
- `*.pdf` -- the source charity documents

The data is UK open data
([source](https://github.com/applicaai/kleister-charity/issues/2)).
