import pandas as pd

from network_wrangler.logger import WranglerLogger

og_stops_df = pd.DataFrame(
    {
        "model_node_id": [1, 2, 3, 2],
        "stop_id": ["a", "b", "c", "extra_b"],
        "name": ["stop1", "stop2", "stop3", "stop222"],
    }
)

expected_stops_df = pd.DataFrame(
    {
        "stop_id": [1, 2, 3],
        "stop_id_GTFS": ["a", "b,extra_b", "c"],
        "name": ["stop1", "stop2", "stop3"],
    }
)


def test_convert_stops_to_wrangler_stops():
    from network_wrangler.models.gtfs.converters import convert_stops_to_wrangler_stops

    out_stops_df = convert_stops_to_wrangler_stops(og_stops_df)
    WranglerLogger.debug(out_stops_df)
    pd.testing.assert_frame_equal(
        out_stops_df.sort_index(axis=1), expected_stops_df.sort_index(axis=1)
    )


og_stop_times_df = pd.DataFrame(
    {
        "stop_id": ["a", "extra_b", "c"],
        "arrival_time": ["12:00:00", "12:30:00", "13:00:00"],
        "departure_time": ["12:00:00", "12:30:00", "13:00:00"],
        "trip_id": [1, 1, 1],
        "stop_sequence": [1, 2, 3],
    }
)

expected_stop_times_df = pd.DataFrame(
    {
        "trip_id": [1, 1, 1],
        "arrival_time": ["12:00:00", "12:30:00", "13:00:00"],
        "departure_time": ["12:00:00", "12:30:00", "13:00:00"],
        "stop_sequence": [1, 2, 3],
        "stop_id": [1, 2, 3],
    }
)


def test_convert_stop_times_to_wrangler_stop_times():
    from network_wrangler.models.gtfs.converters import convert_stop_times_to_wrangler_stop_times

    out_stop_times_df = convert_stop_times_to_wrangler_stop_times(og_stop_times_df, og_stops_df)
    pd.testing.assert_frame_equal(
        out_stop_times_df.sort_index(axis=1), expected_stop_times_df.sort_index(axis=1)
    )
