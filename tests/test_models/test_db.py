"""Tests and Pandera characterization cases for ``DBModelMixin``."""

import copy
from typing import ClassVar

import pandas as pd
import pytest
from pandera import DataFrameModel

from network_wrangler.models._base.db import (
    DBModelMixin,
    ForeignKeyValueError,
    RequiredTableError,
)
from network_wrangler.utils.models import TableValidationError


class MockTableModel_A(DataFrameModel):
    A_ID: int
    name: str

    class Config:
        coerce = True


class MockTableModel_B(DataFrameModel):
    B_ID: int
    a_value: int

    class Config:
        coerce = True
        add_missing_columns = True
        _pk: ClassVar = ["B_ID"]
        _fk: ClassVar = {"a_value": ["table_a", "A_ID"]}


class MockDBModel(DBModelMixin):
    table_names: ClassVar = ["table_a", "table_b"]
    _table_models: ClassVar = {
        "table_a": MockTableModel_A,
        "table_b": MockTableModel_B,
    }


def _valid_table_a():
    return pd.DataFrame({"A_ID": [1, 2, 3], "name": ["a", "b", "c"]})


def _valid_table_b():
    return pd.DataFrame({"B_ID": [4, 5, 6], "a_value": [1, 2, 3]})


def test_validate_db_table():
    db = MockDBModel()
    db.table_a = _valid_table_a()

    # this should validate and be ok
    db.table_b = _valid_table_b()

    with pytest.raises(ForeignKeyValueError):
        db.table_b = pd.DataFrame({"B_ID": [4, 5, 6], "a_value": [3, 4, 5]})

    with pytest.raises(TableValidationError):
        db.table_a = pd.DataFrame({"B_ID": ["hi", "there", "buddy"], "a_value": [3, 4, 5]})


def test_initialize_tables_requires_all_required_tables():
    db = MockDBModel()

    with pytest.raises(RequiredTableError, match="table_b"):
        db.initialize_tables(table_a=_valid_table_a())


def test_initialize_tables_includes_provided_optional_tables():
    class MockTableModel_C(DataFrameModel):
        C_ID: int

        class Config:
            coerce = True

    class OptionalDBModel(DBModelMixin):
        table_names: ClassVar = ["table_a"]
        optional_table_names: ClassVar = ["table_c"]
        _table_models: ClassVar = {
            "table_a": MockTableModel_A,
            "table_c": MockTableModel_C,
        }

    db = OptionalDBModel()
    db.initialize_tables(table_a=_valid_table_a(), table_c=pd.DataFrame({"C_ID": ["9"]}))

    assert db.table_names == ["table_a", "table_c"]
    assert db.table_c["C_ID"].tolist() == [9]
    assert db.modification_version == 2


def test_assignment_increments_modification_version_and_returns_validated_copy():
    db = MockDBModel()
    source = _valid_table_a()

    assert db.modification_version == 0
    db.table_a = source

    assert db.modification_version == 1
    assert db.table_a is not source
    pd.testing.assert_frame_equal(db.table_a, source)


def test_check_table_fks_detects_invalid_in_place_mutation():
    db = MockDBModel()
    db.table_a = _valid_table_a()
    db.table_b = _valid_table_b()

    db.table_b.loc[db.table_b.index[0], "a_value"] = 999

    assert db.check_table_fks("table_b", raise_error=False) is False


def test_in_place_mutation_does_not_increment_modification_version():
    db = MockDBModel()
    db.table_a = _valid_table_a()
    version_before = db.modification_version

    db.table_a.loc[db.table_a.index[0], "name"] = "changed-in-place"

    assert db.modification_version == version_before


def test_database_relationship_metadata_is_derived_from_schema_config():
    assert MockDBModel.fks() == {"table_b": {"a_value": ["table_a", "A_ID"]}}
    assert MockDBModel.fields_as_fks() == {"table_a": {"A_ID": [("table_b", "a_value")]}}


@pytest.mark.xfail(
    strict=True,
    reason="check_fks indexes the summary DataFrame returned by tables_dict as if it were a mapping",
)
def test_check_fks_validates_every_table():
    db = MockDBModel()
    db.table_a = _valid_table_a()
    db.table_b = _valid_table_b()
    db.table_b.loc[db.table_b.index[0], "a_value"] = 999

    assert db.check_fks() is False


def test_deepcopy_has_independent_tables_and_equal_content():
    db = MockDBModel()
    db.table_a = _valid_table_a()
    db.table_b = _valid_table_b()

    cloned = copy.deepcopy(db)

    assert cloned is not db
    assert cloned == db
    assert cloned.table_a is not db.table_a
    assert cloned.table_b is not db.table_b

    cloned.table_a.loc[cloned.table_a.index[0], "name"] = "changed"
    assert db.table_a.loc[db.table_a.index[0], "name"] == "a"
    assert cloned != db


def test_database_hashes_are_stable_for_equal_content_and_change_with_assignment():
    db = MockDBModel()
    db.table_a = _valid_table_a()
    db.table_b = _valid_table_b()
    cloned = copy.deepcopy(db)

    assert db.hash == cloned.hash
    assert hash(db) == hash(cloned)

    cloned.table_a = cloned.table_a.assign(name=["x", "y", "z"])

    assert db.hash != cloned.hash
    assert hash(db) != hash(cloned)


@pytest.mark.xfail(
    strict=True,
    reason="validate_coerce_table catches SchemaErrors but validate_df_to_model raises TableValidationError",
)
def test_converter_runs_after_initial_schema_validation_failure():
    calls = []

    def converter(table, **_kwargs):
        calls.append(True)
        return table.rename(columns={"raw_id": "A_ID"}).assign(name="converted")

    class ConvertingDBModel(DBModelMixin):
        table_names: ClassVar = ["table_a"]
        _table_models: ClassVar = {"table_a": MockTableModel_A}
        _converters: ClassVar = {"table_a": converter}

    db = ConvertingDBModel()
    db.table_a = pd.DataFrame({"raw_id": [1]})

    assert calls == [True]
    assert db.table_a.to_dict("list") == {"A_ID": [1], "name": ["converted"]}


@pytest.mark.xfail(
    strict=True,
    reason="reverse FK checks log and return False but assignment ignores the result",
)
def test_replacing_primary_key_table_rejects_values_still_referenced():
    db = MockDBModel()
    db.table_a = _valid_table_a()
    db.table_b = _valid_table_b()

    with pytest.raises(ForeignKeyValueError):
        db.table_a = pd.DataFrame({"A_ID": [7, 8, 9], "name": ["x", "y", "z"]})
