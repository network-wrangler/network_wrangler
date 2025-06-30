from __future__ import annotations

from datetime import time
from typing import Any, Literal, TypeVar, Union

import pandas as pd

GeoFileTypes = Literal["json", "geojson", "shp", "parquet", "csv", "txt"]

TransitFileTypes = Literal["txt", "csv", "parquet"]

RoadwayFileTypes = Literal["geojson", "shp", "parquet", "json"]

PandasDataFrame = TypeVar("PandasDataFrame", bound=pd.DataFrame)
PandasSeries = TypeVar("PandasSeries", bound=pd.Series)

ForcedStr = Any  # For simplicity, since BeforeValidator is not used here

OneOf = list[list[Union[str, list[str]]]]
ConflictsWith = list[list[str]]
AnyOf = list[list[Union[str, list[str]]]]

Latitude = float
Longitude = float
PhoneNum = str
TimeString = str


# Standalone validator for timespan strings
def validate_timespan_string(value: Any) -> list[str]:
    """Validate that value is a list of exactly 2 time strings in HH:MM or HH:MM:SS format.

    Returns the value if valid, raises ValueError otherwise.
    """
    if not isinstance(value, list):
        msg = "TimespanString must be a list"
        raise ValueError(msg)
    REQUIRED_LENGTH = 2
    if len(value) != REQUIRED_LENGTH:
        msg = f"TimespanString must have exactly {REQUIRED_LENGTH} elements"
        raise ValueError(msg)
    for item in value:
        if not isinstance(item, str):
            msg = "TimespanString elements must be strings"
            raise ValueError(msg)
        import re  # noqa: PLC0415

        if not re.match(r"^(\d+):([0-5]\d)(:[0-5]\d)?$", item):
            msg = f"Invalid time format: {item}"
            raise ValueError(msg)
    return value


TimespanString = list[str]
TimeType = Union[time, str, int]
