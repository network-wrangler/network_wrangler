import re

from typing import Optional, Union, Iterable

import pandera as pa
import pandas as pd

from pandera.engines import pandas_engine


@pandas_engine.Engine.register_dtype
class HttpURL(pandas_engine.NpString):
    name = "http_url"
    str_alias = "http_url"

    def check(
        self,
        pandera_dtype: pa.dtypes.DataType,
        data_container: Optional[pd.Series] = None,
    ) -> Union[bool, Iterable[bool]]:
        correct_type = super().check(pandera_dtype)
        if not correct_type:
            return correct_type

        # Define a regex pattern for HTTP URLs
        pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        return data_container.map(lambda x: pattern.match(x) is not None)

    def __str__(self) -> str:
        return str(self.__class__.__name__)

    def __repr__(self) -> str:
        return f"DataType({self})"
