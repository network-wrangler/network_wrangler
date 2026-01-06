"""Type definitions for GTFS tables."""

import re
from collections.abc import Iterable
from typing import Optional, Union

import pandas as pd
import pandera as pa
from pandera.dtypes import DataType
from pandera.engines import pandas_engine


@pandas_engine.Engine.register_dtype
class HttpURL(pandas_engine.NpString):
    """Pandera DataType for checking if the data is a valid HTTP URL."""

    name = "http_url"
    str_alias = "http_url"

    def check(
        self,
        pandera_dtype: DataType,
        data_container: pd.Series,
    ) -> Union[bool, Iterable[bool]]:
        """Check if the data is a valid HTTP URL."""
        correct_type = super().check(pandera_dtype)
        if not correct_type:
            return correct_type

        # Define a regex pattern for HTTP URLs
        pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        return data_container.map(lambda x: pattern.match(x) is not None)

    def __str__(self) -> str:
        """String representation of the DataType."""
        return str(self.__class__.__name__)

    def __repr__(self) -> str:
        """String representation of the DataType."""
        return f"DataType({self})"
