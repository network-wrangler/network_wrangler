"""Dataframe accessors that allow functions to be called directly on the dataframe."""

import hashlib

import pandas as pd

from ..logger import WranglerLogger
from .data import dict_to_query


@pd.api.extensions.register_dataframe_accessor("dict_query")
class DictQueryAccessor:
    """Query link, node and shape dataframes using project selection dictionary.

    Will overlook any keys which are not columns in the dataframe.

    Usage:

    ```
    selection_dict = {
        "lanes":[1,2,3],
        "name":['6th','Sixth','sixth'],
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
        _selection_dict = {
            k: v for k, v in selection_dict.items() if k in self._obj.columns and v is not None
        }

        if not _selection_dict:
            if return_all_if_none:
                return self._obj
            raise ValueError(f"Relevant part of selection dictionary is empty: {selection_dict}")

        _sel_query = dict_to_query(_selection_dict)
        WranglerLogger.debug(f"_sel_query: \n   {_sel_query}")
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
        """Function to hash the dataframe."""
        _value = str(self._obj.values).encode()
        hash = hashlib.sha1(_value).hexdigest()
        return hash