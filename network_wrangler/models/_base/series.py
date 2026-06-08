import pandera as pa
from pandera.engines import pandas_engine

"""
Time strings in HH:MM or HH:MM:SS format up to 48 hours.
"""
# Use pandas_engine.NpString instead of pa.String to avoid StringDtype compatibility
# issues with numpy.issubdtype in newer pandas versions (2.2+)
# NpString uses object dtype which is compatible with numpy
TimeStrSeriesSchema = pa.SeriesSchema(
    pandas_engine.NpString(),
    pa.Check.str_matches(r"^(?:[0-9]|[0-3][0-9]|4[0-7]):[0-5]\d(?::[0-5]\d)?$|^24:00(?::00)?$"),
    coerce=True,
    name=None,  # Name is set to None to ignore the Series name
)
