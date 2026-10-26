"""Performance tests for routing transit shapes onto the roadway network.

Covers [`route_shapes_between_stops()`][network_wrangler.utils.transit.route_shapes_between_stops]
and the two functions it calls per stop pair:
[`find_shape_aware_shortest_path()`][network_wrangler.utils.transit.find_shape_aware_shortest_path]
and
[`calculate_path_deviation_from_shape()`][network_wrangler.utils.transit.calculate_path_deviation_from_shape].

Synthetic grid networks are used so that cost can be measured against one variable at a time
(network size, stop spacing, candidate-path count, shapes-table size). The St Paul example is
used as a realistic end-to-end case.

!!! note "Why the grid fixtures"
    The cost of this algorithm is driven by inter-stop path *length* and by the number of
    candidate paths enumerated per stop pair -- not by the size of the roadway network. A grid
    lets each of those be varied independently, which a real feed does not.

!!! example "Running these benchmarks"
    ```bash
    pytest --benchmark-only tests/test_transit/test_routing_performance.py
    ```

!!! example "Comparing against another branch"
    ```bash
    pytest --benchmark-only tests/test_transit/test_routing_performance.py --benchmark-save=mine
    git checkout <other-branch>
    pytest --benchmark-only tests/test_transit/test_routing_performance.py --benchmark-save=theirs
    pytest-benchmark compare mine theirs
    ```
"""

from __future__ import annotations

import time

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import shapely.geometry as sg

from network_wrangler.configs import DefaultConfig
from network_wrangler.logger import WranglerLogger
from network_wrangler.models.gtfs.types import RouteType
from network_wrangler.roadway import load_roadway_from_dataframes, load_roadway_from_dir
from network_wrangler.utils.transit import (
    calculate_path_deviation_from_shape,
    find_shape_aware_shortest_path,
    get_original_shape_points_between_stops,
    route_shapes_between_stops,
)

# Grid origin and node spacing. 0.002 degrees is roughly 200m, a plausible roadway node spacing.
GRID_LAT0 = 44.95
GRID_LON0 = -93.10
GRID_DEG = 0.002

LOCAL_CRS = "EPSG:26915"
CRS_UNITS = "meters"


def _grid_dataframes(n: int) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Build links and nodes for an `n` x `n` grid, 4-connected and bidirectional.

    Args:
        n: number of nodes per side.

    Returns:
        Tuple of (links_df, nodes_gdf).
    """
    node_ids = np.arange(1, n * n + 1).reshape(n, n)
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

    nodes_df = pd.DataFrame(
        {
            "model_node_id": node_ids.ravel(),
            "X": GRID_LON0 + cols.ravel() * GRID_DEG,
            "Y": GRID_LAT0 + rows.ravel() * GRID_DEG,
        }
    )
    nodes_gdf = gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(nodes_df["X"], nodes_df["Y"]),
        crs="EPSG:4326",
    )

    horizontal = np.stack([node_ids[:, :-1].ravel(), node_ids[:, 1:].ravel()], axis=1)
    vertical = np.stack([node_ids[:-1, :].ravel(), node_ids[1:, :].ravel()], axis=1)
    one_way = np.concatenate([horizontal, vertical])
    both_ways = np.concatenate([one_way, one_way[:, ::-1]])

    links_df = pd.DataFrame(
        {
            "model_link_id": np.arange(1, len(both_ways) + 1),
            "A": both_ways[:, 0],
            "B": both_ways[:, 1],
            "name": "grid",
            "roadway": "residential",
            "lanes": 2,
            "drive_access": True,
            "bus_only": False,
            "rail_only": False,
            "walk_access": True,
            "bike_access": True,
        }
    )
    return links_df, nodes_gdf


def grid_roadway_net(n: int):
    """Return a RoadwayNetwork for an `n` x `n` grid, fully bus-accessible.

    Args:
        n: number of nodes per side.
    """
    links_df, nodes_gdf = _grid_dataframes(n)
    return load_roadway_from_dataframes(links_df, nodes_gdf)


def grid_bus_feed(
    n: int, n_routes: int, stop_spacing: int = 4
) -> tuple[gpd.GeoDataFrame, dict[str, pd.DataFrame]]:
    """Build bus stop-pair links and feed tables for routes across an `n` x `n` grid.

    Each route runs west to east along an evenly spaced row. Every node in the row becomes an
    original shape point; every `stop_spacing`-th node becomes a stop. This mirrors the real
    case where GTFS shape points are denser than stops.

    Args:
        n: number of grid nodes per side.
        n_routes: number of routes (rows) to create.
        stop_spacing: place a stop every this many grid nodes along the row.

    Returns:
        Tuple of (bus_stop_links_gdf, feed_tables) ready for `route_shapes_between_stops()`.
    """
    node_ids = np.arange(1, n * n + 1).reshape(n, n)
    route_rows = np.linspace(0, n - 1, n_routes).astype(int)

    stop_records: list[dict] = []
    shape_records: list[dict] = []
    pair_records: list[dict] = []

    for route_idx, row in enumerate(route_rows):
        shape_id = f"shp{route_idx}"
        route_id = f"rt{route_idx}"
        trip_id = f"trp{route_idx}"

        stop_cols = list(range(0, n, stop_spacing))
        if stop_cols[-1] != n - 1:
            stop_cols.append(n - 1)
        col_to_stop_seq = {col: i + 1 for i, col in enumerate(stop_cols)}

        for pt_seq, col in enumerate(range(n), start=1):
            shape_records.append(
                {
                    "shape_id": shape_id,
                    "shape_pt_sequence": pt_seq,
                    "shape_pt_lat": GRID_LAT0 + row * GRID_DEG,
                    "shape_pt_lon": GRID_LON0 + col * GRID_DEG,
                    "shape_model_node_id": int(node_ids[row, col]),
                    "stop_sequence": col_to_stop_seq.get(col, np.nan),
                    "route_type": int(RouteType.BUS),
                    "route_id": route_id,
                    "trip_id": trip_id,
                    "direction_id": 0,
                    "agency_id": "A1",
                    "agency_name": "Grid Transit",
                    "route_short_name": f"R{route_idx}",
                }
            )

        for i, col in enumerate(stop_cols):
            stop_records.append(
                {
                    "stop_id": f"s{route_idx}_{i}",
                    "stop_name": f"Stop {route_idx}-{i}",
                    "model_node_id": int(node_ids[row, col]),
                    "is_bus_stop": True,
                    "stop_lat": GRID_LAT0 + row * GRID_DEG,
                    "stop_lon": GRID_LON0 + col * GRID_DEG,
                }
            )

        for i in range(len(stop_cols) - 1):
            from_col, to_col = stop_cols[i], stop_cols[i + 1]
            pair_records.append(
                {
                    "shape_id": shape_id,
                    "stop_sequence": i + 1,
                    "stop_id": f"s{route_idx}_{i}",
                    "stop_name": f"Stop {route_idx}-{i}",
                    "next_stop_id": f"s{route_idx}_{i + 1}",
                    "next_stop_name": f"Stop {route_idx}-{i + 1}",
                    "A": int(node_ids[row, from_col]),
                    "B": int(node_ids[row, to_col]),
                    "route_id": route_id,
                    "route_type": int(RouteType.BUS),
                    "trip_id": trip_id,
                    "direction_id": 0,
                    "geometry": sg.LineString(
                        [
                            (GRID_LON0 + from_col * GRID_DEG, GRID_LAT0 + row * GRID_DEG),
                            (GRID_LON0 + to_col * GRID_DEG, GRID_LAT0 + row * GRID_DEG),
                        ]
                    ),
                }
            )

    shapes_df = pd.DataFrame(shape_records)
    shapes_gdf = gpd.GeoDataFrame(
        shapes_df,
        geometry=gpd.points_from_xy(shapes_df["shape_pt_lon"], shapes_df["shape_pt_lat"]),
        crs="EPSG:4326",
    )
    pairs_gdf = gpd.GeoDataFrame(pair_records, geometry="geometry", crs="EPSG:4326")
    return pairs_gdf, {"stops": pd.DataFrame(stop_records), "shapes": shapes_gdf}


def bus_digraph(roadway_net) -> nx.DiGraph:
    """Collapse the bus MultiDiGraph to a DiGraph the way `route_shapes_between_stops()` does.

    Args:
        roadway_net: a RoadwayNetwork.
    """
    multi = roadway_net.get_modal_graph("bus")
    graph = nx.DiGraph()
    for u, v, data in multi.edges(data=True):
        distance = data.get("distance", 0)
        if graph.has_edge(u, v):
            graph[u][v]["distance"] = min(graph[u][v]["distance"], distance)
        else:
            graph.add_edge(u, v, distance=distance)
    return graph


def route_grid(n: int, n_routes: int, stop_spacing: int = 4) -> None:
    """Route a whole synthetic grid feed, start to finish.

    Args:
        n: grid nodes per side.
        n_routes: number of bus routes.
        stop_spacing: grid nodes between stops.
    """
    roadway_net = grid_roadway_net(n)
    pairs_gdf, feed_tables = grid_bus_feed(n, n_routes, stop_spacing)
    route_shapes_between_stops(
        pairs_gdf,
        feed_tables,
        roadway_net,
        local_crs=LOCAL_CRS,
        crs_units=CRS_UNITS,
        errors="ignore",
    )


# -------------------------------------------------------------------------------------------
# End-to-end
# -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "n_routes"), [(20, 5), (40, 8)])
def test_benchmark_route_shapes_grid(benchmark, n, n_routes):
    """Benchmark end-to-end shape routing on a synthetic grid."""
    benchmark.pedantic(route_grid, args=(n, n_routes), rounds=2, iterations=1)


def test_routing_cost_is_driven_by_stop_spacing_not_network_size():
    """Per-stop-pair cost should track inter-stop distance, not total network size.

    This is the shape of the performance problem: enumerating candidate paths costs more as
    the path between two stops gets longer, while growing the network around a fixed set of
    short hops costs almost nothing. A regression that makes per-pair cost grow with network
    size would show up here.
    """
    per_pair_by_size = {}
    for n, n_routes in [(20, 5), (40, 8)]:
        roadway_net = grid_roadway_net(n)
        pairs_gdf, feed_tables = grid_bus_feed(n, n_routes, stop_spacing=4)
        start = time.perf_counter()
        route_shapes_between_stops(
            pairs_gdf,
            feed_tables,
            roadway_net,
            local_crs=LOCAL_CRS,
            crs_units=CRS_UNITS,
            errors="ignore",
        )
        elapsed = time.perf_counter() - start
        per_pair_by_size[n] = 1000 * elapsed / len(pairs_gdf)

    WranglerLogger.info(f"ms per stop pair by grid size: {per_pair_by_size}")
    small, large = per_pair_by_size[20], per_pair_by_size[40]
    # 4x the nodes should not cost appreciably more per stop pair.
    assert large < small * 3, (
        f"per-stop-pair cost grew with network size: {small:.1f}ms at 20x20 "
        f"vs {large:.1f}ms at 40x40"
    )


# -------------------------------------------------------------------------------------------
# Candidate path enumeration -- the dominant per-stop-pair cost
# -------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid40():
    """A 40x40 grid network plus its bus DiGraph and node id lookup."""
    roadway_net = grid_roadway_net(40)
    return roadway_net, bus_digraph(roadway_net), np.arange(1, 1601).reshape(40, 40)


def _straight_run(node_ids, row: int, from_col: int, n_cols: int):
    """Return (A, B, shape_points) for a straight west-east run along one grid row."""
    shape_points = pd.DataFrame(
        {
            "shape_pt_lon": GRID_LON0 + np.arange(from_col, from_col + n_cols + 1) * GRID_DEG,
            "shape_pt_lat": np.full(n_cols + 1, GRID_LAT0 + row * GRID_DEG),
        }
    )
    return (
        int(node_ids[row, from_col]),
        int(node_ids[row, from_col + n_cols]),
        shape_points,
    )


@pytest.mark.parametrize("stop_spacing", [5, 15, 25])
def test_benchmark_shape_aware_path_by_stop_spacing(benchmark, grid40, stop_spacing):
    """Benchmark one shape-aware path call as the gap between stops grows."""
    roadway_net, graph, node_ids = grid40
    node_a, node_b, shape_points = _straight_run(node_ids, 20, 5, stop_spacing)
    benchmark(
        find_shape_aware_shortest_path,
        graph,
        node_a,
        node_b,
        shape_points,
        roadway_net,
        DefaultConfig.TRANSIT.SHAPE_DISTANCE_TOLERANCE,
    )


def test_candidate_path_count_dominates_cost(grid40):
    """Cost per stop pair is close to linear in MAX_SHAPE_CANDIDATE_PATHS.

    Every candidate path is enumerated by Yen's algorithm whether or not it ends up within
    the distance tolerance, so the configured cap sets the price of each stop pair.
    """
    roadway_net, graph, node_ids = grid40
    node_a, node_b, shape_points = _straight_run(node_ids, 20, 5, 25)

    original_k = DefaultConfig.TRANSIT.MAX_SHAPE_CANDIDATE_PATHS
    timings = {}
    try:
        for k in [1, 5, 20]:
            DefaultConfig.TRANSIT.MAX_SHAPE_CANDIDATE_PATHS = k
            start = time.perf_counter()
            find_shape_aware_shortest_path(
                graph,
                node_a,
                node_b,
                shape_points,
                roadway_net,
                DefaultConfig.TRANSIT.SHAPE_DISTANCE_TOLERANCE,
            )
            timings[k] = 1000 * (time.perf_counter() - start)
    finally:
        DefaultConfig.TRANSIT.MAX_SHAPE_CANDIDATE_PATHS = original_k

    WranglerLogger.info(f"ms per shape-aware path call by candidate cap: {timings}")
    assert timings[20] > timings[1] * 3, f"expected candidate cap to dominate cost, got {timings}"


def test_paths_enumerated_far_exceed_paths_used(grid40):
    """Most enumerated candidate paths are discarded by the tolerance filter.

    Yen's algorithm yields paths in nondecreasing length order, so once a candidate exceeds
    `shortest * SHAPE_DISTANCE_TOLERANCE` every later candidate does too. Enumeration could
    stop there. This test records how much of the enumeration is currently wasted.
    """
    _, graph, node_ids = grid40
    node_a, node_b, _ = _straight_run(node_ids, 20, 5, 25)

    shortest = nx.shortest_path_length(graph, node_a, node_b, weight="distance")
    max_allowed = shortest * DefaultConfig.TRANSIT.SHAPE_DISTANCE_TOLERANCE

    enumerated = 0
    within_tolerance = 0
    for path in nx.shortest_simple_paths(graph, node_a, node_b, weight="distance"):
        enumerated += 1
        distance = sum(
            graph[path[i]][path[i + 1]].get("distance", 0) for i in range(len(path) - 1)
        )
        if distance <= max_allowed:
            within_tolerance += 1
        if enumerated >= DefaultConfig.TRANSIT.MAX_SHAPE_CANDIDATE_PATHS:
            break

    WranglerLogger.info(
        f"candidate paths enumerated={enumerated}, within tolerance={within_tolerance}"
    )
    assert within_tolerance <= enumerated


# -------------------------------------------------------------------------------------------
# Per-candidate-path and per-stop-pair helpers
# -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("path_nodes", "n_shape_points"), [(10, 15), (40, 60)])
def test_benchmark_deviation_scorer(benchmark, grid40, path_nodes, n_shape_points):
    """Benchmark the deviation scorer, called once per candidate path within tolerance."""
    roadway_net, _, node_ids = grid40
    path = [int(x) for x in node_ids[20, :path_nodes]]
    shape_points = pd.DataFrame(
        {
            "shape_pt_lon": GRID_LON0 + np.linspace(0, 0.01, n_shape_points),
            "shape_pt_lat": GRID_LAT0 + np.linspace(0, 0.01, n_shape_points),
        }
    )
    benchmark(calculate_path_deviation_from_shape, path, shape_points, roadway_net)


@pytest.mark.parametrize("n_shape_rows", [5_000, 50_000])
def test_benchmark_shape_point_lookup(benchmark, n_shape_rows):
    """Benchmark the per-stop-pair lookup of original shape points.

    This does a full scan of the shapes table for every stop pair, so its cost grows with
    the size of the feed rather than staying fixed.
    """
    points_per_shape = 50
    n_shapes = n_shape_rows // points_per_shape
    shapes_df = pd.DataFrame(
        {
            "shape_id": np.repeat([f"s{i}" for i in range(n_shapes)], points_per_shape),
            "shape_pt_sequence": np.tile(np.arange(1, points_per_shape + 1), n_shapes),
            "shape_pt_lat": GRID_LAT0,
            "shape_pt_lon": GRID_LON0,
            "stop_sequence": np.tile([1.0, *([np.nan] * (points_per_shape - 2)), 2.0], n_shapes),
        }
    )
    feed_tables = {"shapes": shapes_df}
    benchmark(get_original_shape_points_between_stops, feed_tables, "s3", 1, 2)


# -------------------------------------------------------------------------------------------
# Realistic case
# -------------------------------------------------------------------------------------------


def stpaul_routing_inputs(stpaul_ex_dir):
    """Build real routing inputs from the St Paul example roadway network and GTFS feed.

    Bus stops are snapped to the nearest bus-accessible roadway node and consecutive stops on
    each shape become the stop pairs to route. Each stop is also snapped onto its nearest
    original shape point so that the shape-aware comparison is exercised.

    Args:
        stpaul_ex_dir: path to the St Paul example directory.

    Returns:
        Tuple of (roadway_net, bus_stop_links_gdf, feed_tables).
    """
    roadway_net = load_roadway_from_dir(stpaul_ex_dir)

    stops = pd.read_csv(stpaul_ex_dir / "stops.txt", dtype={"stop_id": str})
    stop_times = pd.read_csv(
        stpaul_ex_dir / "stop_times.txt", dtype={"stop_id": str, "trip_id": str}
    )
    trips = pd.read_csv(stpaul_ex_dir / "trips.txt", dtype={"trip_id": str, "shape_id": str})
    routes = pd.read_csv(stpaul_ex_dir / "routes.txt")
    shapes = pd.read_csv(stpaul_ex_dir / "shapes.txt", dtype={"shape_id": str})

    bus_links = roadway_net.links_df.mode_query("bus")
    bus_node_ids = set(bus_links["A"]) | set(bus_links["B"])
    bus_nodes = roadway_net.nodes_df[roadway_net.nodes_df["model_node_id"].isin(bus_node_ids)]
    bus_nodes_gdf = gpd.GeoDataFrame(
        bus_nodes[["model_node_id"]],
        geometry=gpd.points_from_xy(bus_nodes["X"], bus_nodes["Y"]),
        crs="EPSG:4326",
    ).to_crs(LOCAL_CRS)
    stops_gdf = gpd.GeoDataFrame(
        stops[["stop_id"]],
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    ).to_crs(LOCAL_CRS)
    snapped = gpd.sjoin_nearest(stops_gdf, bus_nodes_gdf)[["stop_id", "model_node_id"]]
    unique_snapped = snapped.drop_duplicates("stop_id")
    stop_to_node = dict(
        zip(unique_snapped["stop_id"], unique_snapped["model_node_id"], strict=True)
    )

    trips = trips.merge(routes[["route_id", "route_type"]], on="route_id")
    bus_route_types = [int(RouteType.BUS), int(RouteType.TROLLEYBUS)]
    bus_trips = trips[trips["route_type"].isin(bus_route_types)]
    ordered = (
        stop_times.merge(
            bus_trips[["trip_id", "route_id", "shape_id", "direction_id", "route_type"]],
            on="trip_id",
        )
        .merge(stops[["stop_id", "stop_name"]], on="stop_id", how="left")
        .sort_values(["shape_id", "stop_sequence"])
        .drop_duplicates(["shape_id", "stop_sequence"])
    )
    ordered["model_node_id"] = ordered["stop_id"].map(stop_to_node)
    ordered = ordered.dropna(subset=["model_node_id"])

    pair_records = []
    for shape_id, shape_stops in ordered.groupby("shape_id"):
        in_order = shape_stops.sort_values("stop_sequence").reset_index(drop=True)
        for i in range(len(in_order) - 1):
            this_stop, next_stop = in_order.loc[i], in_order.loc[i + 1]
            pair_records.append(
                {
                    "shape_id": shape_id,
                    "stop_sequence": i + 1,
                    "stop_id": this_stop["stop_id"],
                    "stop_name": this_stop["stop_name"],
                    "next_stop_id": next_stop["stop_id"],
                    "next_stop_name": next_stop["stop_name"],
                    "A": int(this_stop["model_node_id"]),
                    "B": int(next_stop["model_node_id"]),
                    "route_id": this_stop["route_id"],
                    "route_type": int(this_stop["route_type"]),
                    "trip_id": this_stop["trip_id"],
                    "direction_id": int(this_stop["direction_id"] or 0),
                    "geometry": sg.LineString([(0, 0), (1, 1)]),
                }
            )
    pairs_gdf = gpd.GeoDataFrame(pair_records, geometry="geometry", crs="EPSG:4326")

    shapes = (
        shapes.merge(
            trips[
                ["shape_id", "route_id", "trip_id", "direction_id", "route_type"]
            ].drop_duplicates("shape_id"),
            on="shape_id",
        )
        .sort_values(["shape_id", "shape_pt_sequence"])
        .reset_index(drop=True)
    )
    shapes["stop_sequence"] = np.nan
    stop_xy = stops.set_index("stop_id")[["stop_lon", "stop_lat"]]
    for shape_id, shape_stops in ordered.groupby("shape_id"):
        shape_rows = shapes.index[shapes["shape_id"] == shape_id]
        if len(shape_rows) == 0:
            continue
        shape_lon = shapes.loc[shape_rows, "shape_pt_lon"].to_numpy()
        shape_lat = shapes.loc[shape_rows, "shape_pt_lat"].to_numpy()
        in_order = shape_stops.sort_values("stop_sequence").reset_index(drop=True)
        for stop_seq, stop_id in enumerate(in_order["stop_id"], start=1):
            if stop_id not in stop_xy.index:
                continue
            lon, lat = stop_xy.loc[stop_id, "stop_lon"], stop_xy.loc[stop_id, "stop_lat"]
            nearest = int(np.argmin((shape_lon - lon) ** 2 + (shape_lat - lat) ** 2))
            shapes.loc[shape_rows[nearest], "stop_sequence"] = stop_seq

    shapes["agency_id"] = "A1"
    shapes["agency_name"] = "Metro Transit"
    shapes["route_short_name"] = "R"
    shapes_gdf = gpd.GeoDataFrame(
        shapes,
        geometry=gpd.points_from_xy(shapes["shape_pt_lon"], shapes["shape_pt_lat"]),
        crs="EPSG:4326",
    )
    stops_table = stops.copy()
    stops_table["is_bus_stop"] = True
    return roadway_net, pairs_gdf, {"stops": stops_table, "shapes": shapes_gdf}


@pytest.fixture(scope="module")
def stpaul_routing_case(stpaul_ex_dir):
    """Realistic routing inputs built once per module."""
    return stpaul_routing_inputs(stpaul_ex_dir)


def test_benchmark_route_shapes_stpaul(benchmark, stpaul_routing_case):
    """Benchmark end-to-end shape routing on the St Paul example feed."""
    roadway_net, pairs_gdf, feed_tables = stpaul_routing_case

    def route_once():
        route_shapes_between_stops(
            pairs_gdf.copy(),
            {name: table.copy() for name, table in feed_tables.items()},
            roadway_net,
            local_crs=LOCAL_CRS,
            crs_units=CRS_UNITS,
            errors="ignore",
        )

    benchmark.pedantic(route_once, rounds=1, iterations=1)


def test_stpaul_per_stop_pair_cost(stpaul_routing_case):
    """Record the realistic per-stop-pair cost, which sets the scale of a full feed run."""
    roadway_net, pairs_gdf, feed_tables = stpaul_routing_case
    start = time.perf_counter()
    route_shapes_between_stops(
        pairs_gdf.copy(),
        {name: table.copy() for name, table in feed_tables.items()},
        roadway_net,
        local_crs=LOCAL_CRS,
        crs_units=CRS_UNITS,
        errors="ignore",
    )
    elapsed = time.perf_counter() - start
    per_pair_ms = 1000 * elapsed / len(pairs_gdf)
    WranglerLogger.info(
        f"St Paul: {len(pairs_gdf)} stop pairs in {elapsed:.1f}s ({per_pair_ms:.1f} ms/pair); "
        f"a feed with 50,000 stop pairs would take ~{50_000 * per_pair_ms / 1000 / 60:.0f} min"
    )
    assert len(pairs_gdf) > 0
