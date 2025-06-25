"""Pandas accessor for computing hash of dataframe values."""

import hashlib

import pandas as pd


@pd.api.extensions.register_dataframe_accessor("df_hash")
class DataFrameHashAccessor:
    """Pandas accessor for computing hash of dataframe values."""

    def __init__(self, pandas_obj):
        """Initialize the accessor."""
        self._obj = pandas_obj

    def __call__(self):
        """Compute a hash of the dataframe values that is robust to version differences."""
        # Convert to a more stable representation that's less sensitive to version differences
        # Sort the dataframe to ensure consistent ordering regardless of how it was loaded
        df_sorted = self._obj.sort_index(axis=0).sort_index(axis=1)

        # Convert to a string representation that's more stable
        # Use to_csv with specific parameters to ensure consistency
        csv_string = df_sorted.to_csv(index=False, header=True, sep=",", na_rep="")

        # Compute hash
        return hashlib.sha1(csv_string.encode("utf-8")).hexdigest()
