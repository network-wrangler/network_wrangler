"""
Tests db mixin class for network_wrangler.models._base.db module.

NOTE - THIS SET OF TESTS WAS AUTOGENERATED AND MAY/MAY NOT BE COMPLETE.
"""

import pandas as pd
import pytest
from pandera.errors import SchemaErrors
from network_wrangler.models._base.db import DBModelMixin, RequiredTableError


class MockTableModel:
    def validate(self, table, lazy):
        return table


class MockConverter:
    def __call__(self, table, **kwargs):
        return table


def test_validate_coerce_table():
    db = DBModelMixin()  # Create an instance of the DB class
    table_name = "example_table"
    table = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    # Test when table_name is not in _table_models
    result = db.validate_coerce_table(table_name, table)
    assert result.equals(table)

    # Test when table_name is in _table_models and no converter is available
    db._table_models = {table_name: MockTableModel()}
    with pytest.raises(SchemaErrors):
        db.validate_coerce_table(table_name, table)

    # Test when table_name is in _table_models and a converter is available
    db._converters = {table_name: MockConverter()}
    validated_df = db.validate_coerce_table(table_name, table)
    assert validated_df.equals(table)


def test_validate_coerce_table():
    db = DBModelMixin()  # Create an instance of the DB class
    table_name = "example_table"
    table = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    # Test when table_name is not in _table_models
    result = db.validate_coerce_table(table_name, table)
    assert result.equals(table)

    # Test when table_name is in _table_models and no converter is available
    db._table_models = {table_name: MockTableModel()}
    with pytest.raises(SchemaErrors):
        db.validate_coerce_table(table_name, table)

    # Test when table_name is in _table_models and a converter is available
    db._converters = {table_name: MockConverter()}
    validated_df = db.validate_coerce_table(table_name, table)
    assert validated_df.equals(table)


def test_initialize_tables():
    db = DBModelMixin()  # Create an instance of the DB class
    table1 = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    table2 = pd.DataFrame({"col3": [4, 5, 6], "col4": ["d", "e", "f"]})
    kwargs = {"table1": table1, "table2": table2}

    # Test initialization with all required tables
    db.initialize_tables(**kwargs)
    assert db.table1.equals(table1)
    assert db.table2.equals(table2)

    # Test initialization with missing required tables
    with pytest.raises(RequiredTableError):
        db.initialize_tables(table1=table1)

    # Test initialization with optional tables
    db.optional_table_names = ["table3"]
    table3 = pd.DataFrame({"col5": [7, 8, 9], "col6": ["g", "h", "i"]})
    kwargs["table3"] = table3
    db.initialize_tables(**kwargs)
    assert db.table3.equals(table3)


def test_check_referenced_fk():
    db = DBModelMixin()  # Create an instance of the DB class
    table1 = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    table2 = pd.DataFrame({"col3": [4, 5, 6], "col4": ["d", "e", "f"]})
    kwargs = {"table1": table1, "table2": table2}
    db.initialize_tables(**kwargs)

    # Test checking referenced foreign keys with valid values
    assert db.check_referenced_fk("table1", "col1")

    # Test checking referenced foreign keys with invalid values
    table2.loc[0, "col3"] = 7
    assert not db.check_referenced_fk("table1", "col1")


def test_check_referenced_fks():
    db = DBModelMixin()  # Create an instance of the DB class
    table1 = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    table2 = pd.DataFrame({"col3": [4, 5, 6], "col4": ["d", "e", "f"]})
    kwargs = {"table1": table1, "table2": table2}
    db.initialize_tables(**kwargs)

    # Test checking referenced foreign keys with valid values
    assert db.check_referenced_fks("table1", table1)

    # Test checking referenced foreign keys with invalid values
    table2.loc[0, "col3"] = 7
    assert not db.check_referenced_fks("table1", table1)


def test_check_table_fks():
    db = DBModelMixin()  # Create an instance of the DB class
    table1 = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    table2 = pd.DataFrame({"col3": [4, 5, 6], "col4": ["d", "e", "f"]})
    kwargs = {"table1": table1, "table2": table2}
    db.initialize_tables(**kwargs)

    # Test checking foreign keys with valid values
    assert db.check_table_fks("table1", table1)

    # Test checking foreign keys with invalid values
    table1.loc[0, "col1"] = 7
    assert not db.check_table_fks("table1", table1)
