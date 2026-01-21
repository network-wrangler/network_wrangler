"""Dataframe accessors that allow functions to be called directly on the dataframe."""

import hashlib

import pandas as pd

from ..errors import SelectionError
from ..logger import WranglerLogger
from .data import dict_to_query, isin_dict


@pd.api.extensions.register_dataframe_accessor("dict_query")
class DictQueryAccessor:
    """Query link, node and shape dataframes using project selection dictionary.

    Will overlook any keys which are not columns in the dataframe.

    Usage:

    ```
    selection_dict = {
        "lanes": [1, 2, 3],
        "name": ["6th", "Sixth", "sixth"],
        "drive_access": 1,
    }
    selected_links_df = links_df.dict_query(selection_dict)
    ```

    """

    def __init__(self, pandas_obj):
        """Initialization function for the dictionary query accessor."""
        self._obj = pandas_obj

    def __call__(self, selection_dict: dict, return_all_if_none: bool = False):
        """Queries the dataframe using the selection dictionary.

        Args:
            selection_dict (dict): _description_
            return_all_if_none (bool, optional): If True, will return entire df if dict has
                 no values. Defaults to False.
        """
        _not_selection_keys = ["modes", "all", "ignore_missing"]
        _selection_dict = {
            k: v
            for k, v in selection_dict.items()
            if k not in _not_selection_keys and v is not None
        }
        missing_columns = [k for k in _selection_dict if k not in self._obj.columns]
        if missing_columns:
            msg = f"Selection fields not found in dataframe: {missing_columns}"
            raise SelectionError(msg)

        if not _selection_dict:
            if return_all_if_none:
                return self._obj
            msg = f"Relevant part of selection dictionary is empty: {selection_dict}"
            raise SelectionError(msg)

        _sel_query = dict_to_query(_selection_dict)
        # WranglerLogger.debug(f"_sel_query: \n   {_sel_query}")
        _df = self._obj.query(_sel_query, engine="python")

        if len(_df) == 0:
            WranglerLogger.warning(
                f"No records found in df \
                  using selection: {selection_dict}"
            )
        return _df


@pd.api.extensions.register_dataframe_accessor("df_hash")
class dfHash:
    """Creates a dataframe hash that is compatable with geopandas and various metadata.

    Definitely not the fastest, but she seems to work where others have failed.
    """

    def __init__(self, pandas_obj):
        """Initialization function for the dataframe hash."""
        self._obj = pandas_obj

    def __call__(self):
        """Function to hash the dataframe with version-robust computation."""
        # Convert to a more stable representation that's less sensitive to version differences
        # Sort the dataframe to ensure consistent ordering regardless of how it was loaded
        df_sorted = self._obj.sort_index(axis=0).sort_index(axis=1)

        # Use a more stable string representation that's less sensitive to version differences
        # Convert to numpy array and then to string, which is more consistent across versions
        _value = str(df_sorted.values.tolist()).encode()
        hash = hashlib.sha1(_value).hexdigest()
        return hash


@pd.api.extensions.register_dataframe_accessor("isin_dict")
class Isin_dict:
    """Faster implimentation of isin for querying dataframes with dictionary."""

    def __init__(self, pandas_obj):
        """Initialization function for the dataframe hash."""
        self._obj = pandas_obj

    def __call__(self, d: dict, **kwargs) -> pd.DataFrame:
        """Function to perform the faster dictionary isin()."""
        return isin_dict(self._obj, d, **kwargs)
