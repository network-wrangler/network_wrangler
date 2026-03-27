"""Tests db mixin class for network_wrangler.models._base.db module."""

from typing import ClassVar

import pandas as pd
import pytest
from pandera import DataFrameModel

from network_wrangler.models._base.db import DBModelMixin, ForeignKeyValueError
from network_wrangler.utils.models import TableValidationError


class MockTableModel_A(DataFrameModel):
    A_ID: int
    name: str

    class Config:
        coerce = True

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


def test_validate_db_table():
    db = MockDBModel()
    db.table_a = pd.DataFrame({"A_ID": [1, 2, 3], "name": ["a", "b", "c"]})

    # this should validate and be ok
    db.table_b = pd.DataFrame({"B_ID": [4, 5, 6], "a_value": [1, 2, 3]})

    with pytest.raises(ForeignKeyValueError):
        db.table_b = pd.DataFrame({"B_ID": [4, 5, 6], "a_value": [3, 4, 5]})

    with pytest.raises(TableValidationError):
        db.table_a = pd.DataFrame({"B_ID": ["hi", "there", "buddy"], "a_value": [3, 4, 5]})
