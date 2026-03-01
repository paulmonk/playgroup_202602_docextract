"""Tests for the extraction pipeline. No LLM calls, all mocked."""

from __future__ import annotations

import pytest

from commands import pipeline

# Shared test fixture
_SAMPLE_ROW = ("doc.pdf", "keys", "djvu_text", "tess411_text", "tess_march_text", "combined_text")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("djvu2hocr", "djvu_text"),
        ("tesseract411", "tess411_text"),
        ("tesseract_march2020", "tess_march_text"),
        ("combined", "combined_text"),
    ],
)
def test_select_text_returns_correct_column(source, expected):
    assert pipeline.select_text(_SAMPLE_ROW, source=source) == expected


def test_select_text_unknown_source_raises():
    with pytest.raises(ValueError, match="Unknown source"):
        pipeline.select_text(_SAMPLE_ROW, source="nonexistent")


def test_select_text_short_row_raises():
    short_row = ("doc.pdf", "keys")
    with pytest.raises(ValueError, match="columns"):
        pipeline.select_text(short_row, source="combined")


def test_select_text_default_source_is_combined():
    assert pipeline.select_text(_SAMPLE_ROW) == "combined_text"


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        # Money formatting
        ("income_annually_in_british_pounds", "255653", "255653.00"),
        ("income_annually_in_british_pounds", "255,653", "255653.00"),
        ("income_annually_in_british_pounds", "255653.00", "255653.00"),
        ("spending_annually_in_british_pounds", "18797665.0", "18797665.00"),
        # Date normalisation
        ("report_date", "2015-12-31", "2015-12-31"),
        ("report_date", "31/12/2015", "2015-12-31"),
        ("report_date", "31-12-2015", "2015-12-31"),
        ("report_date", "31.12.2015", "2015-12-31"),
        # Post town: uppercase, spaces to underscores
        ("address__post_town", "Newcastle upon Tyne", "NEWCASTLE_UPON_TYNE"),
        ("address__post_town", "LONDON", "LONDON"),
        ("address__post_town", "Bourne End", "BOURNE_END"),
        # Postcode: uppercase, spaces to underscores
        ("address__postcode", "SY3 7PQ", "SY3_7PQ"),
        ("address__postcode", "sy3 7pq", "SY3_7PQ"),
        ("address__postcode", "EC3N 1DY", "EC3N_1DY"),
        # Street line: spaces to underscores
        ("address__street_line", "58 TRINITY STREET", "58_TRINITY_STREET"),
        ("address__street_line", "34-38 BRENKLEY WAY", "34-38_BRENKLEY_WAY"),
        # Charity name: preserve case, spaces to underscores
        ("charity_name", "The Sanata Charitable Trust", "The_Sanata_Charitable_Trust"),
        ("charity_name", "Mental Health Concern", "Mental_Health_Concern"),
        # Charity number: strip leading zeros
        ("charity_number", "0300703", "300703"),
        ("charity_number", "1132766", "1132766"),
    ],
)
def test_normalise_for_output(key, value, expected):
    assert pipeline.normalise_for_output(key, value=value) == expected


def test_normalise_money_with_pound_sign():
    result = pipeline.normalise_for_output(
        "income_annually_in_british_pounds", value="\u00a3255,653"
    )
    assert result == "255653.00"


def _validate(fields: dict[str, str]) -> dict[str, str]:
    """Helper: validate fields through the Pydantic model."""
    return pipeline.CharityExtraction.model_validate(fields).to_fields()


@pytest.mark.parametrize(
    "value",
    ["1132766", "300703", "12345", "519332"],
)
def test_valid_charity_number(value):
    result = _validate({"charity_number": value})
    assert "charity_number" in result


@pytest.mark.parametrize(
    "value",
    ["1234", "12345678", "abc", "", "12.34"],
)
def test_invalid_charity_number_stripped(value):
    result = _validate({"charity_number": value})
    assert "charity_number" not in result


@pytest.mark.parametrize(
    "value",
    ["2015-12-31", "1990-01-01", "2030-12-31"],
)
def test_valid_report_date(value):
    result = _validate({"report_date": value})
    assert "report_date" in result


@pytest.mark.parametrize(
    "value",
    ["1989-12-31", "2031-01-01", "not-a-date", "31/12/2015", ""],
)
def test_invalid_report_date_stripped(value):
    result = _validate({"report_date": value})
    assert "report_date" not in result


@pytest.mark.parametrize(
    "value",
    ["SY3 7PQ", "EC3N 1DY", "N16 5PQ", "NR14 7DU", "PE31 6RH", "SG11 2DY"],
)
def test_valid_postcode(value):
    result = _validate({"address__postcode": value})
    assert "address__postcode" in result


@pytest.mark.parametrize(
    "value",
    ["INVALID", "123", ""],
)
def test_invalid_postcode_stripped(value):
    result = _validate({"address__postcode": value})
    assert "address__postcode" not in result


@pytest.mark.parametrize(
    "value",
    ["255653.00", "1.00", "18797665.00", "100"],
)
def test_valid_money(value):
    result = _validate({"income_annually_in_british_pounds": value})
    assert "income_annually_in_british_pounds" in result


@pytest.mark.parametrize(
    "value",
    ["-100", "0", "abc", ""],
)
def test_invalid_money_stripped(value):
    result = _validate({"income_annually_in_british_pounds": value})
    assert "income_annually_in_british_pounds" not in result


def test_unrecognised_fields_dropped():
    result = _validate({"charity_name": "Test Charity", "unknown_field": "foo"})
    assert result == {"charity_name": "Test Charity"}


def test_mix_of_valid_and_invalid():
    fields = {
        "charity_number": "1132766",
        "report_date": "not-a-date",
        "charity_name": "Test Charity",
    }
    result = _validate(fields)
    assert "charity_number" in result
    assert "report_date" not in result
    assert "charity_name" in result


def test_parse_llm_response_valid_json():
    raw = '{"charity_number": "1132766", "report_date": "2015-12-31"}'
    result = pipeline.parse_llm_response(raw)
    assert result == {"charity_number": "1132766", "report_date": "2015-12-31"}


def test_parse_llm_response_json_in_triple_backticks():
    raw = 'Here is the data:\n```json\n{"charity_number": "1132766"}\n```'
    result = pipeline.parse_llm_response(raw)
    assert result == {"charity_number": "1132766"}


def test_parse_llm_response_malformed_json_returns_empty():
    raw = "this is not json at all"
    result = pipeline.parse_llm_response(raw)
    assert result == {}


def test_parse_llm_response_extra_fields_ignored():
    raw = '{"charity_number": "1132766", "unknown_field": "foo", "report_date": "2015-12-31"}'
    result = pipeline.parse_llm_response(raw)
    assert "unknown_field" not in result
    assert result == {"charity_number": "1132766", "report_date": "2015-12-31"}


def test_parse_llm_response_missing_fields_ok():
    raw = '{"charity_number": "1132766"}'
    result = pipeline.parse_llm_response(raw)
    assert result == {"charity_number": "1132766"}


def test_parse_llm_response_empty_string_returns_empty():
    result = pipeline.parse_llm_response("")
    assert result == {}


def test_parse_llm_response_numeric_values_converted_to_strings():
    raw = '{"charity_number": 1132766, "income_annually_in_british_pounds": 255653.00}'
    result = pipeline.parse_llm_response(raw)
    assert result["charity_number"] == "1132766"
    assert result["income_annually_in_british_pounds"] == "255653.0"


def test_parse_llm_response_json_array_returns_empty():
    raw = '[{"charity_number": "1132766"}]'
    result = pipeline.parse_llm_response(raw)
    assert result == {}


def test_parse_llm_response_json_embedded_in_prose():
    raw = (
        'Based on my analysis:\n{"charity_number": "264289",'
        ' "report_date": "2014-12-31"}\nKey notes: ...'
    )
    result = pipeline.parse_llm_response(raw)
    assert result == {"charity_number": "264289", "report_date": "2014-12-31"}


def test_format_row_sorted_key_value_pairs():
    fields = {
        "charity_number": "1132766",
        "charity_name": "The Sanata Charitable Trust",
        "address__post_town": "Shrewsbury",
    }
    result = pipeline.format_row(fields)
    parts = result.split("\t")
    keys = [p.split("=")[0] for p in parts]
    assert keys == sorted(keys)


def test_format_row_empty_fields_produce_empty_string():
    assert pipeline.format_row({}) == ""


def test_format_row_normalisation_applied():
    fields = {"address__post_town": "Bourne End", "income_annually_in_british_pounds": "41721"}
    result = pipeline.format_row(fields)
    assert "address__post_town=BOURNE_END" in result
    assert "income_annually_in_british_pounds=41721.00" in result


def test_format_row_single_field():
    result = pipeline.format_row({"charity_number": "0300703"})
    assert result == "charity_number=300703"


def test_format_row_missing_fields_omitted():
    fields = {"charity_number": "1132766"}
    result = pipeline.format_row(fields)
    assert "report_date" not in result
    assert "charity_number=1132766" in result


def test_format_row_full_row_matches_expected_format():
    fields = {
        "address__post_town": "SHREWSBURY",
        "address__postcode": "SY3 7PQ",
        "address__street_line": "58 TRINITY STREET",
        "charity_name": "The Sanata Charitable Trust",
        "charity_number": "1132766",
        "income_annually_in_british_pounds": "255653.00",
        "report_date": "2015-12-31",
        "spending_annually_in_british_pounds": "258287.00",
    }
    result = pipeline.format_row(fields)
    expected = "\t".join(
        [
            "address__post_town=SHREWSBURY",
            "address__postcode=SY3_7PQ",
            "address__street_line=58_TRINITY_STREET",
            "charity_name=The_Sanata_Charitable_Trust",
            "charity_number=1132766",
            "income_annually_in_british_pounds=255653.00",
            "report_date=2015-12-31",
            "spending_annually_in_british_pounds=258287.00",
        ]
    )
    assert result == expected


def test_extract_empty_llm_response(mocker):
    mock_llm = mocker.patch("commands.pipeline.llm_openrouter")
    mock_llm.call_llm.return_value = None
    result = pipeline.extract("some text", model="test-model")
    assert result == {}


def test_extract_llm_returns_valid_json(mocker):
    mock_llm = mocker.patch("commands.pipeline.llm_openrouter")
    mock_llm.call_llm.return_value = '{"charity_number": "1132766"}'
    result = pipeline.extract("some text", model="test-model")
    assert result == {"charity_number": "1132766"}


def test_extract_llm_returns_garbage(mocker):
    mock_llm = mocker.patch("commands.pipeline.llm_openrouter")
    mock_llm.call_llm.return_value = "I cannot extract any fields from this document."
    result = pipeline.extract("some text", model="test-model")
    assert result == {}


def test_normalise_for_output_charity_number_non_numeric():
    result = pipeline.normalise_for_output("charity_number", value="not-a-number")
    assert result == "not-a-number"


def test_normalise_for_output_strips_newlines():
    result = pipeline.normalise_for_output("charity_name", value="Foo\nBar\rBaz")
    assert "\n" not in result
    assert "\r" not in result
    assert result == "Foo_Bar_Baz"


def test_normalise_for_output_strips_smart_quotes():
    result = pipeline.normalise_for_output("charity_name", value="\u2018Foo\u2019s \u201cBar\u201d")
    assert result == "'Foo's_\"Bar\""


def test_extract_from_triple_backticks_with_lang_tag():
    text = '```json\n{"key": "value"}\n```'
    result = pipeline._extract_from_triple_backticks(text)
    assert result == '{"key": "value"}'


def test_extract_from_triple_backticks_no_lang_tag():
    text = "```\nhello world\n```"
    result = pipeline._extract_from_triple_backticks(text)
    assert result == "hello world"


def test_extract_from_triple_backticks_none_when_no_backticks():
    assert pipeline._extract_from_triple_backticks("just plain text") is None


def test_extract_from_triple_backticks_strips_comment_markers():
    text = "```\n/// some comment\n```"
    result = pipeline._extract_from_triple_backticks(text)
    assert result == "some comment"


def test_extract_from_triple_backticks_strips_double_slash():
    text = "```\n// some comment\n```"
    result = pipeline._extract_from_triple_backticks(text)
    assert result == "some comment"


def test_extract_from_triple_backticks_uses_last_block():
    text = '```\nfirst\n```\nsome text\n```\n{"second": true}\n```'
    result = pipeline._extract_from_triple_backticks(text)
    assert result == '{"second": true}'


def test_parse_json_object_from_mixed_text_skips_array():
    """The parser should skip JSON objects inside arrays."""
    text = 'Some text [{"charity_number": "123"}] more text'
    result = pipeline._parse_json_object_from_mixed_text(text)
    assert result is None


def test_parse_json_object_from_mixed_text_finds_standalone():
    """The parser should find a standalone JSON object in mixed text."""
    text = 'Before {"charity_number": "456"} after'
    result = pipeline._parse_json_object_from_mixed_text(text)
    assert result == {"charity_number": "456"}


def test_parse_json_object_from_mixed_text_skips_array_finds_later_object():
    """When an array comes first, the parser should find a later standalone object."""
    text = 'List: [{"a": 1}] but also {"charity_number": "789"}'
    result = pipeline._parse_json_object_from_mixed_text(text)
    assert result == {"charity_number": "789"}


def test_extract_json_dict_non_json_backtick_falls_through():
    """When backtick content is not valid JSON, fall through to mixed-text parser."""
    raw = '```\nnot json at all\n```\n{"charity_number": "111"}'
    result = pipeline._extract_json_dict(raw)
    assert result == {"charity_number": "111"}


def test_parse_llm_response_all_fields_invalid_logs_warning():
    """When JSON parses but all fields fail validation, should return empty dict."""
    raw = '{"charity_number": "12345678", "report_date": "not-a-date"}'
    result = pipeline.parse_llm_response(raw)
    assert result == {}


def test_write_output(tmp_path):
    out = tmp_path / "out.tsv"
    pipeline.write_output(["line1", "line2"], output_path=str(out))
    lines = out.read_text().splitlines()
    assert lines == ["line1", "line2"]
