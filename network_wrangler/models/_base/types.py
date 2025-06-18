from __future__ import annotations

from datetime import time
from typing import Any, List, Literal, TypeVar, Union

import pandas as pd
from pydantic import Field

GeoFileTypes = Literal["json", "geojson", "shp", "parquet", "csv", "txt"]

TransitFileTypes = Literal["txt", "csv", "parquet"]

RoadwayFileTypes = Literal["geojson", "shp", "parquet", "json"]

PandasDataFrame = TypeVar("PandasDataFrame", bound=pd.DataFrame)
PandasSeries = TypeVar("PandasSeries", bound=pd.Series)

ForcedStr = Any  # For simplicity, since BeforeValidator is not used here

OneOf = List[List[Union[str, List[str]]]]
ConflictsWith = List[List[str]]
AnyOf = List[List[Union[str, List[str]]]]

Latitude = float
Longitude = float
PhoneNum = str
TimeString = str


# Standalone validator for timespan strings
def validate_timespan_string(value: Any) -> List[str]:
    """Validate that value is a list of exactly 2 time strings in HH:MM or HH:MM:SS format.
    Returns the value if valid, raises ValueError otherwise.
    """
    if not isinstance(value, list):
        raise ValueError("TimespanString must be a list")
    if len(value) != 2:
        raise ValueError("TimespanString must have exactly 2 elements")
    for item in value:
        if not isinstance(item, str):
            raise ValueError("TimespanString elements must be strings")
        import re

        if not re.match(r"^(\d+):([0-5]\d)(:[0-5]\d)?$", item):
            raise ValueError(f"Invalid time format: {item}")
    return value


TimespanString = List[str]
TimeType = Union[time, str, int]
