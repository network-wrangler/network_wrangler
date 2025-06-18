"""Functions related to parsing and comparing time objects and series.

Internal function terminology for timespan scopes:

- `matching`: a scope that could be applied for a given timespan combination.
    This includes the default timespan as well as scopes wholely contained within.
- `overlapping`: a timespan that fully or partially overlaps a given timespan.
    This includes the default timespan, all `matching` timespans and all timespans where
    at least one minute overlap.
- `conflicting`: a timespan that is overlapping but not matching. By definition default
     scope values are not conflicting.
- `independent` a timespan that is not overlapping.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional, Union

import pandas as pd
from pydantic import validate_call

from ..logger import WranglerLogger
from ..models._base.series import TimeStrSeriesSchema
from ..models._base.types import TimespanString, TimeString


class TimespanDfQueryError(Exception):
    """Error for timespan query errors."""


@validate_call(config={"arbitrary_types_allowed": True})
def str_to_time(time_str: TimeString, base_date: Optional[date] = None) -> datetime:
    """Convert TimeString (HH:MM<:SS>) to datetime object.

    If HH > 24, will subtract 24 to be within 24 hours. Timespans will be treated as the next day.

    Args:
        time_str: TimeString in HH:MM:SS or HH:MM format.
        base_date: optional date to base the datetime on. Defaults to None.
            If not provided, will use today.
    """
    # Set the base date to today if not provided
    if base_date is None:
        base_date = date.today()

    # Split the time string to extract hours, minutes, and seconds
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2]) if len(parts) == 3 else 0  # noqa: PLR2004

    if hours >= 24:  # noqa: PLR2004
        add_days = hours // 24
        base_date += timedelta(days=add_days)
        hours -= 24 * add_days

    # Create a time object with the adjusted hours, minutes, and seconds
    adjusted_time = datetime.strptime(f"{hours:02}:{minutes:02}:{seconds:02}", "%H:%M:%S").time()

    # Combine the base date with the adjusted time and add the extra days if needed
    combined_datetime = datetime.combine(base_date, adjusted_time)

    return combined_datetime


def _all_str_to_time_series(
    time_str_s: pd.Series, base_date: Optional[Union[pd.Series, date]] = None
) -> pd.Series:
    """Assume all are strings and convert to datetime objects."""
    # check strings are in the correct format, leave existing date times alone
    TimeStrSeriesSchema.validate(time_str_s)

    # Set the base date to today if not provided
    if base_date is None:
        base_date = pd.Series([date.today()] * len(time_str_s), index=time_str_s.index)
    elif isinstance(base_date, date):
        base_date = pd.Series([base_date] * len(time_str_s), index=time_str_s.index)
    elif len(base_date) != len(time_str_s):
        msg = "base_date must be the same length as time_str_s"
        WranglerLogger.error(msg)
        raise ValueError(msg)

    # Split the time string to extract hours, minutes, and seconds
    time_parts = time_str_s.str.split(":", expand=True).astype(int)
    hours = time_parts[0]
    minutes = time_parts[1]
    seconds = time_parts[2] if time_parts.shape[1] == 3 else 0  # noqa: PLR2004

    if (hours >= 24).any():  # noqa: PLR2004
        hours[hours >= 24] -= 24  # noqa: PLR2004

    # Combine the base date with the adjusted time and add the extra days if needed
    combined_datetimes = (
        pd.to_datetime(base_date)
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
        + pd.to_timedelta(seconds, unit="s")
    )

    return combined_datetimes


def str_to_time_series(
    time_str_s: pd.Series, base_date: Optional[Union[pd.Series, date]] = None
) -> pd.Series:
    """Convert mixed panda series datetime and TimeString (HH:MM<:SS>) to datetime object.

    If HH > 24, will subtract 24 to be within 24 hours. Timespans will be treated as the next day.

    Args:
        time_str_s: Pandas Series of TimeStrings in HH:MM:SS or HH:MM format.
        base_date: optional date to base the datetime on. Defaults to None.
            If not provided, will use today. Can be either a single instance or a series of
            same length as time_str_s
    """
    # check strings are in the correct format, leave existing date times alone
    is_string = time_str_s.apply(lambda x: isinstance(x, str))
    time_strings = time_str_s[is_string]
    result = time_str_s.copy()
    if is_string.any():
        result[is_string] = _all_str_to_time_series(time_strings, base_date)
    result = result.astype("datetime64[ns]")
    return result


@validate_call(config={"arbitrary_types_allowed": True})
def str_to_time_list(timespan: list[TimeString]) -> list[datetime]:
    """Convert list of TimeStrings (HH:MM<:SS>) to list of datetime.time objects."""
    timespan_dt: list[datetime] = list(map(str_to_time, timespan))
    if not is_increasing(timespan_dt):
        timespan_dt = [timespan_dt[0], timespan_dt[1] + timedelta(days=1)]
        WranglerLogger.warning(
            f"Timespan is not in increasing order: {timespan}.\
            End time will be treated as next day."
        )
    return timespan_dt


@validate_call(config={"arbitrary_types_allowed": True})
def timespan_str_list_to_dt(timespans: list[TimespanString]) -> list[list[datetime]]:
    """Convert list of TimespanStrings to list of datetime.time objects."""
    return [str_to_time_list(ts) for ts in timespans]


@validate_call(config={"arbitrary_types_allowed": True})
def dt_to_seconds_from_midnight(dt: datetime) -> int:
    """Convert a datetime object to the number of seconds since midnight."""
    return round((dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds())


@validate_call(config={"arbitrary_types_allowed": True})
def str_to_seconds_from_midnight(time_str: TimeString) -> int:
    """Convert a TimeString (HH:MM<:SS>) to the number of seconds since midnight."""
    dt = str_to_time(time_str)
    return dt_to_seconds_from_midnight(dt)


@validate_call(config={"arbitrary_types_allowed": True})
def seconds_from_midnight_to_str(seconds: int) -> TimeString:
    """Convert the number of seconds since midnight to a TimeString (HH:MM)."""
    return str(timedelta(seconds=seconds))


@validate_call(config={"arbitrary_types_allowed": True})
def filter_df_to_overlapping_timespans(
    orig_df: pd.DataFrame,
    query_timespans: list[TimespanString],
) -> pd.DataFrame:
    """Filters dataframe for entries that have any overlap with ANY of the given query timespans.

    If the end time is less than the start time, it is assumed to be the next day.

    Args:
        orig_df: dataframe to query timespans for with `start_time` and `end_time` fields.
        query_timespans: List of a list of TimespanStr of format ['HH:MM','HH:MM'] to query orig_df
            for overlapping records.
    """
    if "start_time" not in orig_df.columns or "end_time" not in orig_df.columns:
        msg = "DataFrame must have 'start_time' and 'end_time' columns"
        WranglerLogger.error(msg)
        raise TimespanDfQueryError(msg)

    mask = pd.Series([False] * len(orig_df), index=orig_df.index)
    for query_timespan in query_timespans:
        q_start_time, q_end_time = str_to_time_list(query_timespan)
        end_time_s = orig_df["end_time"]
        end_time_s.loc[orig_df["end_time"] < orig_df["start_time"]] += pd.Timedelta(days=1)
        this_ts_mask = (orig_df["start_time"] < q_end_time) & (q_start_time < end_time_s)
        mask |= this_ts_mask
    return orig_df.loc[mask]


def calc_overlap_duration_with_query(
    start_time_s: pd.Series[datetime],
    end_time_s: pd.Series[datetime],
    start_time_q: datetime,
    end_time_q: datetime,
) -> pd.Series[timedelta]:
    """Calculate the overlap series of start and end times and a query start and end times.

    Args:
        start_time_s: Series of start times to calculate overlap with.
        end_time_s: Series of end times to calculate overlap with.
        start_time_q: Query start time to calculate overlap with.
        end_time_q: Query end time to calculate overlap with.
    """
    overlap_start = start_time_s.combine(start_time_q, max)
    overlap_end = end_time_s.combine(end_time_q, min)
    overlap_duration_s = (overlap_end - overlap_start).dt.total_seconds() / 60

    return overlap_duration_s


@validate_call(config={"arbitrary_types_allowed": True})
def filter_df_to_max_overlapping_timespans(
    orig_df: pd.DataFrame,
    query_timespan: list[TimeString],
    strict_match: bool = False,
    min_overlap_minutes: int = 1,
    keep_max_of_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Filters dataframe for entries that have maximum overlap with the given query timespan.

    If the end time is less than the start time, it is assumed to be the next day.

    Args:
        orig_df: dataframe to query timespans for with `start_time` and `end_time` fields.
        query_timespan: TimespanString of format ['HH:MM','HH:MM'] to query orig_df for overlapping
            records.
        strict_match: boolean indicating if the returned df should only contain
            records that fully contain the query timespan. If set to True, min_overlap_minutes
            does not apply. Defaults to False.
        min_overlap_minutes: minimum number of minutes the timespans need to overlap to keep.
            Defaults to 1.
        keep_max_of_cols: list of fields to return the maximum value of overlap for.  If None,
            will return all overlapping time periods. Defaults to `['model_link_id']`
    """
    if keep_max_of_cols is None:
        keep_max_of_cols = ["model_link_id"]
    if "start_time" not in orig_df.columns or "end_time" not in orig_df.columns:
        msg = "DataFrame must have 'start_time' and 'end_time' columns"
        WranglerLogger.error(msg)
        raise TimespanDfQueryError(msg)
    q_start, q_end = str_to_time_list(query_timespan)

    real_end = orig_df["end_time"]
    real_end.loc[orig_df["end_time"] < orig_df["start_time"]] += pd.Timedelta(days=1)

    orig_df["overlap_duration"] = calc_overlap_duration_with_query(
        orig_df["start_time"],
        real_end,
        q_start,
        q_end,
    )
    if strict_match:
        overlap_df = orig_df.loc[(orig_df.start_time <= q_start) & (real_end >= q_end)]
    else:
        overlap_df = orig_df.loc[orig_df.overlap_duration > min_overlap_minutes]
    WranglerLogger.debug(f"overlap_df: \n{overlap_df}")
    if keep_max_of_cols:
        # keep only the maximum overlap
        idx = overlap_df.groupby(keep_max_of_cols)["overlap_duration"].idxmax()
        overlap_df = overlap_df.loc[idx]
    return overlap_df


def convert_timespan_to_start_end_dt(timespan_s: pd.Serie[str]) -> pd.DataFrame:
    """Convert a timespan string ['12:00','14:00] to start_time & end_time datetime cols in df."""
    start_time = timespan_s.apply(lambda x: str_to_time(x[0]))
    end_time = timespan_s.apply(lambda x: str_to_time(x[1]))
    return pd.DataFrame({"start_time": start_time, "end_time": end_time})


@validate_call
def dt_overlap_duration(timedelta1: timedelta, timedelta2: timedelta) -> timedelta:
    """Check if two timespans overlap and return the amount of overlap.

    If the end time is less than the start time, it is assumed to be the next day.
    """
    if timedelta1.end_time < timedelta1.start_time:
        timedelta1 = timedelta1 + timedelta(days=1)
    if timedelta2.end_time < timedelta2.start_time:
        timedelta2 = timedelta2 + timedelta(days=1)
    overlap_start = max(timedelta1.start_time, timedelta2.start_time)
    overlap_end = min(timedelta1.end_time, timedelta2.end_time)
    overlap_duration = max(overlap_end - overlap_start, timedelta(0))
    return overlap_duration


@validate_call
def dt_contains(timespan1: list[datetime], timespan2: list[datetime]) -> bool:
    """Check timespan1 inclusively contains timespan2.

    If the end time is less than the start time, it is assumed to be the next day.

    Args:
        timespan1 (list[time]): The first timespan represented as a list containing the start
            time and end time.
        timespan2 (list[time]): The second timespan represented as a list containing the start
            time and end time.

    Returns:
        bool: True if the first timespan contains the second timespan, False otherwise.
    """
    start_time_dt, end_time_dt = timespan1

    if end_time_dt < start_time_dt:
        end_time_dt = end_time_dt + timedelta(days=1)

    start_time_dt2, end_time_dt2 = timespan2

    if end_time_dt2 < start_time_dt2:
        end_time_dt2 = end_time_dt2 + timedelta(days=1)

    return (start_time_dt <= start_time_dt2) and (end_time_dt >= end_time_dt2)


@validate_call(config={"arbitrary_types_allowed": True})
def dt_overlaps(timespan1: list[datetime], timespan2: list[datetime]) -> bool:
    """Check if two timespans overlap.

    If the end time is less than the start time, it is assumed to be the next day.

    `overlapping`: a timespan that fully or partially overlaps a given timespan.
    This includes and all timespans where at least one minute overlap.
    """
    time1_start, time1_end = timespan1
    time2_start, time2_end = timespan2

    if time1_end < time1_start:
        time1_end += timedelta(days=1)
    if time2_end < time2_start:
        time2_end += timedelta(days=1)

    return (time1_start < time2_end) and (time2_start < time1_end)


def timespans_overlap(timespan1: list[TimespanString], timespan2: list[TimespanString]) -> bool:
    """Check if two timespan strings overlap.

    `overlapping`: a timespan that fully or partially overlaps a given timespan.
    This includes and all timespans where at least one minute overlap.
    """
    timespan1 = str_to_time_list(timespan1)
    timespan2 = str_to_time_list(timespan2)
    return dt_overlaps(timespan1, timespan2)


@validate_call
def filter_dt_list_to_overlaps(timespans: list[list[datetime]]) -> list[list[datetime]]:
    """Filter a list of timespans to only include those that overlap.

    `overlapping`: a timespan that fully or partially overlaps a given timespan.
    This includes and all timespans where at least one minute overlap.
    """
    overlaps = []
    for i in range(len(timespans)):
        for j in range(i + 1, len(timespans)):
            if dt_overlaps(timespans[i], timespans[j]):
                overlaps += [timespans[i], timespans[j]]

    # remove dupes
    return list(map(list, set(map(tuple, overlaps))))


def dt_list_overlaps(timespans: list[list[datetime]]) -> bool:
    """Check if any of the timespans overlap.

    `overlapping`: a timespan that fully or partially overlaps a given timespan.
    This includes and all timespans where at least one minute overlap.
    """
    return bool(filter_dt_list_to_overlaps(timespans))


def duration_dt(start_time_dt: datetime, end_time_dt: datetime) -> timedelta:
    """Returns a datetime.timedelta object representing the duration of the timespan.

    If end_time is less than start_time, the duration will assume that it crosses over
    midnight.
    """
    if end_time_dt < start_time_dt:
        return timedelta(
            hours=24 - start_time_dt.hour + end_time_dt.hour,
            minutes=end_time_dt.minute - start_time_dt.minute,
            seconds=end_time_dt.second - start_time_dt.second,
        )
    return end_time_dt - start_time_dt


def format_seconds_to_legible_str(seconds: int) -> str:
    """Formats seconds into a human-friendly string for log files."""
    if seconds < 60:  # noqa: PLR2004
        return f"{int(seconds)} seconds"
    if seconds < 3600:  # noqa: PLR2004
        return f"{int(seconds // 60)} minutes"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours} hours and {minutes} minutes"


def is_increasing(datetimes: list[datetime]) -> bool:
    """Check if a list of datetime objects is increasing in time."""
    return all(datetimes[i] <= datetimes[i + 1] for i in range(len(datetimes) - 1))
