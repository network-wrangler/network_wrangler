"""Utils for converting original gtfs to wrangler gtfs."""

from pathlib import Path

import pandas as pd


def convert_gtfs_to_wrangler_gtfs(gtfs_path: Path, wrangler_path: Path) -> None:
    """Converts a GTFS feed to a Wrangler GTFS feed.

    Args:
        gtfs_path: Path to GTFS feed.
        wrangler_path: Path to save Wrangler GTFS feed.
    """
    gtfs_path = Path(gtfs_path)
    gtfs_stops_df = pd.read_csv(gtfs_path / "stops.txt")
    gtfs_stoptimes_df = pd.read_csv(gtfs_path / "stop_times.txt")

    wr_stoptimes_df = convert_stop_times_to_wrangler_stop_times(gtfs_stoptimes_df, gtfs_stops_df)
    wr_stops_df = convert_stops_to_wrangler_stops(gtfs_stops_df)

    wrangler_path = Path(wrangler_path)
    if not wrangler_path.exists():
        wrangler_path.mkdir(parents=True)
    wr_stops_df.to_csv(wrangler_path / "stops.txt", index=False)
    wr_stoptimes_df.to_csv(wrangler_path / "stop_times.txt", index=False)


def convert_stops_to_wrangler_stops(stops_df: pd.DataFrame) -> pd.DataFrame:
    """Converts a stops.txt file to a Wrangler stops.txt file.

    Creates table that is unique to each model_node_id.
    Takes first instance of value for all attributes except stop_id.
    Aggregates stop_id into a comma-separated string and renames to gtfs_stop_id.
    Renames model_node_id to stop_id.

    Example usage:

    ```python
    import pandas as pd
    from network_wrangler.models.gtfs.converters import convert_stops_to_wrangler_stops

    in_f = "network_wrangler/examples/stpaul/gtfs/stops.txt"
    stops_df = pd.read_csv(in_f)
    wr_stops_df = convert_stops_to_wrangler_stops(stops_df)
    wr_stops_df.to_csv("wr_stops.txt", index=False)
    ```
    Args:
        stops_df: stops.txt file as a pandas DataFrame.

    Returns:
        Wrangler stops.txt file as a pandas DataFrame.
    """
    wr_stops_df = stops_df.groupby("model_node_id").first().reset_index()
    wr_stops_df = wr_stops_df.drop(columns=["stop_id"])
    # if stop_id is an int, convert to string
    if stops_df["stop_id"].dtype == "int64":
        stops_df["stop_id"] = stops_df["stop_id"].astype(str)
    gtfs_stop_id = stops_df.groupby("model_node_id").stop_id.apply(",".join).reset_index()
    wr_stops_df["gtfs_stop_id"] = gtfs_stop_id["stop_id"]
    wr_stops_df = wr_stops_df.rename(columns={"model_node_id": "stop_id"})
    return wr_stops_df


def convert_stop_times_to_wrangler_stop_times(
    gtfs_stop_times_df: pd.DataFrame, gtfs_stops_df: pd.DataFrame
) -> pd.DataFrame:
    """Converts a stop_times.txt file to a Wrangler stop_times.txt file.

    Replaces stop_id with model_node_id from stops.txt, making sure that if there are duplicate
    model_node_ids for each stop_id, the correct model_node_id is used.

    Args:
        gtfs_stop_times_df: stop_times.txt file as a pandas DataFrame.
        gtfs_stops_df: stops.txt file as a pandas DataFrame

    Returns:
        Wrangler stop_times.txt file as a pandas DataFrame.
    """
    wr_stop_times_df = gtfs_stop_times_df.merge(
        gtfs_stops_df[["stop_id", "model_node_id"]], on="stop_id", how="left"
    )
    wr_stop_times_df = wr_stop_times_df.drop(columns=["stop_id"])
    wr_stop_times_df = wr_stop_times_df.rename(columns={"model_node_id": "stop_id"})
    return wr_stop_times_df
