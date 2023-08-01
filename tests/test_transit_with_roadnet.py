import os

from projectcard import read_card
from network_wrangler import RoadwayNetwork
from network_wrangler import TransitNetwork

"""
Run just the tests labeled transit using `pytest -v -m transit`
"""

STPAUL_DIR = os.path.join(os.getcwd(), "examples", "stpaul")
SCRATCH_DIR = os.path.join(os.getcwd(), "scratch")


def test_set_roadnet(request):
    print("\n--Starting:", request.node.name)

    road_net = RoadwayNetwork.read(
        links_file=os.path.join(STPAUL_DIR, "link.json"),
        nodes_file=os.path.join(STPAUL_DIR, "node.geojson"),
        shapes_file=os.path.join(STPAUL_DIR, "shape.geojson"),
    )
    transit_net = TransitNetwork.read(STPAUL_DIR)
    transit_net.set_roadnet(road_net)

    print("--Finished:", request.node.name)


def test_project_card(request):
    print("\n--Starting:", request.node.name)

    road_net = RoadwayNetwork.read(
        links_file=os.path.join(STPAUL_DIR, "link.json"),
        nodes_file=os.path.join(STPAUL_DIR, "node.geojson"),
        shapes_file=os.path.join(STPAUL_DIR, "shape.geojson"),
    )
    transit_net = TransitNetwork.read(STPAUL_DIR)
    transit_net.road_net = road_net
    project_card_path = os.path.join(
        STPAUL_DIR, "project_cards", "12_transit_shape_change.yml"
    )
    project_card = read_card(project_card_path)
    transit_net = transit_net.apply_transit_feature_change(
        transit_net.get_selection(project_card.facility).selected_trips,
        project_card.properties,
    )

    # Shapes
    result = transit_net.feed.shapes[transit_net.feed.shapes["shape_id"] == "2940002"][
        "shape_model_node_id"
    ].tolist()
    answer = [
        "37582",
        "37574",
        "4761",
        "4763",
        "4764",
        "98429",
        "45985",
        "57483",
        "126324",
        "57484",
        "150855",
        "11188",
        "84899",
        "46666",
        "46665",
        "46663",
        "81820",
        "76167",
        "77077",
        "68609",
        "39425",
        "62146",
        "41991",
        "70841",
        "45691",
        "69793",
        "45683",
        "45685",
        "7688",
        "45687",
        "100784",
        "100782",
        "45688",
        "37609",
        "19077",
        "38696",
    ]
    assert result == answer

    # Stops
    result = transit_net.feed.stop_times[
        transit_net.feed.stop_times["trip_id"] == "14944022-JUN19-MVS-BUS-Weekday-01"
    ]["stop_id"].tolist()
    result_tail = result[-5:]
    answer_tail = ["17013", "17010", "17009", "17006", "17005"]
    assert result_tail == answer_tail

    print("--Finished:", request.node.name)


def test_wo_existing(request):
    print("\n--Starting:", request.node.name)

    road_net = RoadwayNetwork.read(
        links_file=os.path.join(STPAUL_DIR, "link.json"),
        nodes_file=os.path.join(STPAUL_DIR, "node.geojson"),
        shapes_file=os.path.join(STPAUL_DIR, "shape.geojson"),
    )
    transit_net = TransitNetwork.read(STPAUL_DIR)
    transit_net.road_net = road_net

    transit_net = transit_net.apply_transit_feature_change(
        trip_ids=transit_net.get_selection(
            {"trip_id": ["14986385-JUN19-MVS-BUS-Weekday-01"]}
        ).selected_trips,
        properties=[{"property": "routing", "set": [75318]}],
    )

    # Stops
    result = transit_net.feed.stop_times[
        transit_net.feed.stop_times["trip_id"] == "14986385-JUN19-MVS-BUS-Weekday-01"
    ]["stop_id"].tolist()
    answer = ["2609"]  # first matching stop_id in stops.txt
    assert result == answer

    print("--Finished:", request.node.name)


def test_select_transit_features_by_nodes(request):
    print("\n--Starting:", request.node.name)

    transit_net = TransitNetwork.read(STPAUL_DIR)

    # Any nodes
    trip_ids = transit_net.get_selection(
        {"nodes": ["75520", "66380", "57530"]}
    ).selected_trips
    print(trip_ids)
    assert set(trip_ids) == set(
        [
            "14941148-JUN19-MVS-BUS-Weekday-01",
            "14941151-JUN19-MVS-BUS-Weekday-01",
            "14941153-JUN19-MVS-BUS-Weekday-01",
            "14941163-JUN19-MVS-BUS-Weekday-01",
            "14944379-JUN19-MVS-BUS-Weekday-01",
            "14944386-JUN19-MVS-BUS-Weekday-01",
            "14944413-JUN19-MVS-BUS-Weekday-01",
            "14944416-JUN19-MVS-BUS-Weekday-01",
        ]
    )

    # All nodes
    trip_ids = transit_net.get_selection(
        {"nodes": ["75520", "66380"], "require_all": True}
    ).selected_trips
    assert set(trip_ids) == set(
        [
            "14941148-JUN19-MVS-BUS-Weekday-01",
            "14941151-JUN19-MVS-BUS-Weekday-01",
            "14941153-JUN19-MVS-BUS-Weekday-01",
            "14941163-JUN19-MVS-BUS-Weekday-01",
        ]
    )

    print("--Finished:", request.node.name)
