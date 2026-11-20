"""Tests and Pandera characterization cases for ``network_wrangler.utils.models``."""

import copy

import pandas as pd
import pyarrow as pa
import pytest
from pandera import DataFrameModel, Field
from pandera.errors import SchemaErrors
from pandera.typing import DataFrame, Series
from pydantic import BaseModel, ConfigDict, ValidationError

from network_wrangler.utils.models import (
    TableValidationError,
    _convert_string_dtype_to_object,
    coerce_extra_fields_to_type_in_df,
    default_from_datamodel,
    empty_df_from_datamodel,
    fill_df_with_defaults_from_model,
    identify_model,
    order_fields_from_data_model,
    submodel_fields_in_model,
    validate_call_pyd,
    validate_df_to_model,
)


class Submodel(BaseModel):
    model_config = ConfigDict(extra="allow")

    submofield1: int


class SampleModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    field1: str
    field2: int
    submo: Submodel


class CharacterizationTable(DataFrameModel):
    """Small schema exercising the Pandera features used by helper functions."""

    ident: Series[int] = Field(coerce=True, unique=True)
    label: Series[str] = Field(default="unknown", nullable=False)
    count: Series[int] = Field(default=0, coerce=True, nullable=False)
    enabled: Series[bool] = Field(default=False, coerce=True, nullable=False)
    optional: Series[str] | None = Field(default=None, nullable=True, coerce=True)

    class Config:
        add_missing_columns = True
        coerce = True


class AlternateCharacterizationTable(DataFrameModel):
    """Second schema used to expose current ``identify_model`` behavior."""

    code: Series[str] = Field(coerce=True)


class DefaultCharacterizationTable(DataFrameModel):
    """Schema used to characterize falsey, null, and timestamp defaults."""

    zero: Series[int] = Field(default=0)
    disabled: Series[bool] = Field(default=False)
    empty_text: Series[str] = Field(default="")
    timestamp: Series[pd.Timestamp] = Field(default=pd.Timestamp("2026-01-02"))
    optional: Series[object] | None = Field(default=None, nullable=True)


class MutableDefaultCharacterizationTable(DataFrameModel):
    """Schema exposing how the helper handles a mutable column default."""

    payload: Series[object] = Field(default=[])


@validate_call_pyd
def _decorated_table_call(df: DataFrame[CharacterizationTable], repetitions: int) -> tuple:
    """Return values after ``validate_call_pyd`` processes annotations."""
    return df, repetitions


@validate_call_pyd
def _decorated_optional_table_call(
    df: DataFrame[CharacterizationTable] | None,
) -> DataFrame[CharacterizationTable] | None:
    """Exercise Pandera table annotations nested in a union and in the return type."""
    return df


def test_submodel_fields_in_model():
    submo = submodel_fields_in_model(SampleModel)
    assert submo == ["submo"]


def test_coerce_extra_fields_to_type_in_df():
    # Create a sample DataFrame
    df = pd.DataFrame(
        {
            "field1": ["value1", "value2"],
            "field2": [1, 2],
            "field3": [True, False],
            "field4": [1, 2],
            "field5": ["1", "2"],
        }
    )

    # Create a sample data instance
    data = SampleModel(
        field1="value3",
        field2=3,
        field3=True,
        submo=Submodel(submofield1=3, field4="4"),
        field5=[5, 7],
    )
    # Call the function
    coerced_data = coerce_extra_fields_to_type_in_df(data, SampleModel, df)

    # Check if the extra field 'field3' is coerced to the type in the DataFrame
    assert isinstance(coerced_data.field3, bool)

    # Check if submodel also coerced
    assert isinstance(coerced_data.submo.field4, int)

    # Check if list values are coerced
    assert coerced_data.field5 == ["5", "7"]


def test_validate_df_to_model_preserves_table_contract():
    """Validation copies data and preserves index, attrs, extras, and their order."""
    source = pd.DataFrame(
        {"extra": ["x", "y"], "ident": ["1", "2"]},
        index=pd.Index([10, 20], name="source_row"),
    )
    source.attrs = {"nested": {"values": [1, 2]}}
    source_before = source.copy(deep=True)
    attrs_before = copy.deepcopy(source.attrs)

    result = validate_df_to_model(source, CharacterizationTable)

    assert result is not source
    pd.testing.assert_frame_equal(source, source_before)
    assert result.index.equals(source.index)
    assert result.index.name == "source_row"
    assert result.attrs == attrs_before
    assert result.attrs is not source.attrs
    assert result.attrs["nested"] is not source.attrs["nested"]
    assert list(result.columns) == [
        "extra",
        "ident",
        "label",
        "count",
        "enabled",
    ]
    assert result.to_dict("list") == {
        "extra": ["x", "y"],
        "ident": [1, 2],
        "label": ["unknown", "unknown"],
        "count": [0, 0],
        "enabled": [False, False],
    }

    result.loc[10, "ident"] = 99
    assert source.loc[10, "ident"] == "1"


@pytest.mark.parametrize("dtype", [object, pd.StringDtype(), pd.ArrowDtype(pa.string())])
def test_convert_string_extension_dtypes_to_object(dtype):
    source = pd.DataFrame({"text": pd.Series(["a", None], dtype=dtype)})

    result = _convert_string_dtype_to_object(source)

    assert result is not source
    assert result["text"].dtype == object
    assert source["text"].dtype == dtype


def test_validate_df_to_model_collects_independent_column_failures(tmp_path):
    invalid = pd.DataFrame(
        {
            "ident": ["not-an-int", "not-an-int"],
            "label": [None, None],
            "count": ["not-an-int", "not-an-int"],
        }
    )

    with pytest.raises(TableValidationError) as exc_info:
        validate_df_to_model(
            invalid,
            CharacterizationTable,
            output_file=tmp_path / "aggregated_failures.csv",
        )

    failure_columns = set(exc_info.value.__cause__.failure_cases["column"].dropna())
    assert {"ident", "count"}.issubset(failure_columns)


def test_validate_df_to_model_aggregates_small_errors_without_file(tmp_path):
    output_file = tmp_path / "small_failures.csv"
    invalid = pd.DataFrame({"ident": ["not-an-int"]})

    with pytest.raises(TableValidationError) as exc_info:
        validate_df_to_model(invalid, CharacterizationTable, output_file=output_file)

    assert str(exc_info.value) == "Validation to CharacterizationTable failed."
    assert isinstance(exc_info.value.__cause__, SchemaErrors)
    assert not output_file.exists()
    failure_cases = exc_info.value.__cause__.failure_cases
    assert "ident" in failure_cases["column"].astype(str).tolist()


def test_validate_df_to_model_writes_large_failure_case_csv(tmp_path):
    output_file = tmp_path / "large_failures.csv"
    invalid = pd.DataFrame({"ident": [f"bad-{i}" for i in range(6)]})

    with pytest.raises(TableValidationError) as exc_info:
        validate_df_to_model(invalid, CharacterizationTable, output_file=output_file)

    assert isinstance(exc_info.value.__cause__, SchemaErrors)
    assert output_file.exists()
    written = pd.read_csv(output_file)
    assert len(written) >= 6
    assert {"schema_context", "column", "check", "failure_case", "index"}.issubset(written.columns)


def test_defaults_and_field_order_helpers():
    source = pd.DataFrame(
        {
            "extra_b": [1],
            "enabled": [None],
            "ident": [1],
            "label": [None],
            "count": [None],
            "optional": [pd.NA],
            "extra_a": [2],
        }
    )

    assert default_from_datamodel(CharacterizationTable, "count") == 0
    assert default_from_datamodel(CharacterizationTable, "enabled") is False
    assert default_from_datamodel(CharacterizationTable, "label") == "unknown"
    assert default_from_datamodel(CharacterizationTable, "optional") is None
    assert default_from_datamodel(CharacterizationTable, "missing") is None

    filled = fill_df_with_defaults_from_model(source.copy(), CharacterizationTable)
    assert not bool(filled.loc[0, "enabled"])
    assert filled.loc[0, "count"] == 0
    assert filled.loc[0, "label"] == "unknown"
    assert filled.loc[0, "optional"] is None

    ordered = order_fields_from_data_model(source, CharacterizationTable)
    assert list(ordered.columns) == [
        "ident",
        "label",
        "count",
        "enabled",
        "optional",
        "extra_b",
        "extra_a",
    ]

    without_optional = order_fields_from_data_model(
        source.drop(columns="optional"), CharacterizationTable
    )
    assert "optional" not in without_optional


def test_fill_defaults_preserves_falsey_values_and_normalizes_nulls():
    source = pd.DataFrame(
        {
            "zero": [pd.NA],
            "disabled": [pd.NA],
            "empty_text": [pd.NA],
            "timestamp": [pd.NaT],
            "optional": [pd.NA],
        }
    )

    result = fill_df_with_defaults_from_model(source, DefaultCharacterizationTable)

    assert result.loc[0, "zero"] == 0
    assert result.loc[0, "disabled"] is False
    assert result.loc[0, "empty_text"] == ""
    assert result.loc[0, "timestamp"] == pd.Timestamp("2026-01-02")
    assert result.loc[0, "optional"] is None


@pytest.mark.xfail(
    strict=True,
    reason="fill_df_with_defaults_from_model passes a list to Series.fillna, which rejects it",
)
def test_fill_defaults_creates_independent_mutable_values():
    source = pd.DataFrame({"payload": [None, None]})

    result = fill_df_with_defaults_from_model(source, MutableDefaultCharacterizationTable)

    assert result["payload"].tolist() == [[], []]
    assert result.loc[0, "payload"] is not result.loc[1, "payload"]


def test_empty_df_from_datamodel_has_declared_columns():
    result = empty_df_from_datamodel(CharacterizationTable)

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == list(CharacterizationTable.__fields__)


def test_identify_model_matches_first_dataframe_and_pydantic_model():
    assert identify_model(pd.DataFrame({"ident": [1]}), [CharacterizationTable]) is (
        CharacterizationTable
    )
    assert (
        identify_model(
            {"field1": "text", "field2": 1, "submo": {"submofield1": 2}},
            [SampleModel],
        )
        is SampleModel
    )


def test_identify_model_reports_no_pydantic_model_match():
    with pytest.raises(TableValidationError, match="isn't consistant"):
        identify_model({"field1": "missing-other-fields"}, [SampleModel])


@pytest.mark.xfail(
    strict=True,
    reason="identify_model does not catch TableValidationError and stops after its first table model",
)
def test_identify_model_tries_later_dataframe_models():
    result = identify_model(
        pd.DataFrame({"code": ["valid-for-second-model"]}),
        [CharacterizationTable, AlternateCharacterizationTable],
    )
    assert result is AlternateCharacterizationTable


def test_validate_call_pyd_preserves_table_and_validates_non_table_arguments():
    source = pd.DataFrame({"unvalidated": [object()]})

    returned, repetitions = _decorated_table_call(source, "2")

    assert returned is source
    assert repetitions == 2

    with pytest.raises(ValidationError):
        _decorated_table_call(source, "not-an-int")


def test_validate_call_pyd_handles_optional_table_and_return_annotations():
    source = pd.DataFrame({"unvalidated": [object()]})

    assert _decorated_optional_table_call(source) is source
    assert _decorated_optional_table_call(None) is None
