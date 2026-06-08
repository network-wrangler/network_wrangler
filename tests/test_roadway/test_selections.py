"""Tests related to selecting roadway features.

To run with print statments, use `pytest -s tests/test_roadway/test_selections.py`
"""

import copy

import pandas as pd
import pytest
from projectcard import read_card

from network_wrangler import WranglerLogger
from network_wrangler.models.projects.roadway_selection import RoadwaySelectionFormatError
from network_wrangler.utils.data import dict_to_query


def test_dfhash(request, stpaul_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 5]})
    hash1 = df.df_hash()
    hash2 = df.df_hash()
    WranglerLogger.debug(f"Simple DF\nhash1: {hash1}\nhash2: {hash2}")
    assert hash1 == hash2

    df = stpaul_net.nodes_df
    hash1 = df.df_hash()
    hash2 = df.df_hash()
    WranglerLogger.debug(f"Full Nodes\nhash1: {hash1}\nhash2: {hash2}")
    assert hash1 == hash2
    WranglerLogger.info(f"--Finished: {request.node.name}")


TEST_SELECTIONS = [
    {  # SELECTION 0
        "links": {
            "name": ["6th", "Sixth", "sixth"],
        },
        "from": {"osm_node_id": "187899923"},
        "to": {"osm_node_id": "187865924"},
    },
    {  # SELECTION 1
        "links": {
            "name": ["Lafayette"],
        },
        "from": {"osm_node_id": "2292977517"},
        "to": {"osm_node_id": "507951637"},
    },
    {  # SELECTION 2
        "links": {
            "name": ["University Ave"],
            "lanes": [1],
        },
        "from": {"osm_node_id": "716319401"},
        "to": {"model_node_id": "62153"},
    },
    {  # SELECTION 3
        "links": {
            "name": ["I 35E"],
        },
        "from": {"osm_node_id": "954746969"},
        "to": {"osm_node_id": "3071141242"},
    },
    {  # SELECTION 4 FIXME
        "links": {
            "osm_link_id": ["221685893"],
        },
        "from": {"model_node_id": "131209"},
        "to": {"model_node_id": "131221"},
    },
    {  # SELECTION 5
        "links": {
            "model_link_id": ["390239", "391206", "281", "1464"],
            "lanes": [1, 2],
        },
    },
    {  # SELECTION 6
        "links": {
            "all": True,
            "lanes": [1, 2],
        },
    },
    {  # SELECTION 7
        "links": {
            "all": True,
        },
    },
    {  # SELECTION 8
        "links": {
            "name": ["Valley Street"],
            "modes": ["walk"],
        },
        "from": {"model_node_id": 174762},
        "to": {"model_node_id": 43041},
    },
    {  # SELECTION 8
        "links": {
            "name": ["Minnehaha"],
            "modes": ["drive"],
        },
    },
]


answer_selected_links = [
    [134543, 85185, 154004],  # SELECTION 0
    [386035, 401018, 401019],  # SELECTION 1
    [412924, 389361],  # SELECTION 2
    [381412, 392837, 394194, 394196, 391146],  # SELECTION 3
    [294513, 294518, 294532],  # SELECTION 4
    [281, 1464],  # SELECTION 5
    None,  # SELECTION 6 - all links
    None,  # SELECTION 7 - all links with some features
    [460228, 481940],  # SELECTION 8 - Valley Street Pedestrian Ways
    [
        7194,
        7196,
        7298,
        7300,
        8481,
        8483,
        8495,
        8496,
        14895,
        14896,
        86566,
        86567,
        94717,
        94719,
        101600,
        101601,
        107314,
        107315,
        109422,
        109423,
        111886,
        111890,
        111893,
        111894,
        111896,
        111897,
        111899,
        111900,
        111902,
        111906,
        111908,
        111910,
        111912,
        111915,
        111916,
        111922,
        111923,
        111927,
        111929,
        111931,
        111932,
        111933,
        111935,
        111936,
        132888,
        132889,
        155432,
        155433,
        156611,
        156612,
        163088,
        163089,
        163522,
        163523,
        171952,
        171955,
        171956,
        171962,
        171964,
        171966,
        171967,
        171969,
        171970,
        171974,
        171975,
        171978,
        171979,
        275245,
    ],  # SELECTION 9 - Minnehaha
]


@pytest.mark.parametrize(
    ("selection", "answer"), zip(TEST_SELECTIONS, answer_selected_links, strict=False)
)
def test_select_roadway_features(request, selection, answer, stpaul_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    net = stpaul_net
    WranglerLogger.info(f"Selecting--->{selection}")
    WranglerLogger.info(f"Num Net Selections: {len(net._selections)}")
    _selection = net.get_selection(selection)
    _show_f = ["A", "B", "name", "osm_link_id", "model_link_id", "lanes"]
    selected_link_indices = _selection.selected_links
    WranglerLogger.info(f"{len(_selection.selected_links)} links selected")
    if _selection.selection_method == "segment":
        WranglerLogger.info(f"Segment Path: \n{_selection.segment.segment_nodes}")
        WranglerLogger.info(f"Segment Links: \n{_selection.segment.segment_links_df[_show_f]}")
    WranglerLogger.info("Selected Links")

    if len(selected_link_indices) < 10:
        WranglerLogger.info(f"Selected Links: \n{_selection.selected_links_df[_show_f]}")

    if answer:
        WranglerLogger.info(f"Answer Links: {answer}")
        assert set(selected_link_indices) == set(answer)

    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_select_roadway_features_from_projectcard(request, stpaul_net, stpaul_ex_dir):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    net = stpaul_net

    # selection is a facility segment
    project_card_name = "road.prop_change.multiple.yml"
    _expected_answer = [134543, 85185, 154004]

    project_card_path = stpaul_ex_dir / "project_cards" / project_card_name
    project_card = read_card(project_card_path)

    _facility = project_card.roadway_property_change["facility"]
    _selection = net.get_selection(_facility)
    selected_link_idx = _selection.selected_links
    WranglerLogger.debug(f"Features selected: {len(selected_link_idx)}")

    assert set(_selection.selected_links) == set(_expected_answer)

    WranglerLogger.info(f"--Finished: {request.node.name}")


variable_queries = [
    ({"v": "lanes", "category": "sov", "timespan": ["12:00", "12:30"]}, 3),
    ({"v": "lanes", "timespan": ["12:00", "12:30"]}, 3),
    ({"v": "lanes", "timespan": ["7:00", "9:00"]}, 2),
    ({"v": "ML_price", "category": "sov", "timespan": ["7:00", "9:00"]}, 1.5),
    (
        {
            "v": "ML_price",
            "categories": ["hov3", "hov2"],
            "timespan": ["7:00", "9:00"],
        },
        1,
    ),
]


@pytest.fixture(scope="module")
def ml_card(stpaul_ex_dir):
    project_card_path = stpaul_ex_dir / "project_cards" / "road.managed_lanes.whole_facility.yml"
    return read_card(project_card_path)


@pytest.fixture(scope="module")
def ml_net(stpaul_net, ml_card):
    net = copy.deepcopy(stpaul_net)
    net = net.apply(ml_card)
    return net


@pytest.mark.parametrize("variable_query", variable_queries)
def test_query_roadway_property_by_time_group(request, variable_query, ml_net, ml_card):
    WranglerLogger.info(f"--Starting: {request.node.name}")

    _query, _answer = variable_query

    v_series = ml_net.get_property_by_timespan_and_group(
        _query["v"],
        category=_query.get("category"),
        timespan=_query.get("timespan"),
    )

    _selected_links = ml_net.get_selection(
        ml_card.roadway_property_change["facility"]
    ).selected_links
    WranglerLogger.debug(f"QUERY: \n{_query}")
    WranglerLogger.debug(f"EXPECTED ANSWER: {_answer}")
    WranglerLogger.debug(f"QUERY RESULT: \n{v_series.loc[_selected_links]}")
    WranglerLogger.debug(f"NET: \n{ml_net.links_df.loc[_selected_links[0]][_query['v']]}")

    assert (v_series.loc[_selected_links, _query["v"]] == _answer).all()

    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_get_modal_network(request, stpaul_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    from network_wrangler.params import MODES_TO_NETWORK_LINK_VARIABLES

    net = stpaul_net
    mode = "transit"
    _links_df = net.links_df.mode_query(mode)

    test_links_of_selection = _links_df["model_link_id"].tolist()
    WranglerLogger.debug(f"TEST - Number of selected links: {len(test_links_of_selection)}")

    control_links_of_selection = []
    for m in MODES_TO_NETWORK_LINK_VARIABLES[mode]:
        if m not in net.links_df.columns:
            continue
        control_links_of_selection.extend(net.links_df.loc[net.links_df[m], "model_link_id"])
    WranglerLogger.debug(f"CONTROL - Number of selected links: {len(control_links_of_selection)}")

    all_model_link_ids = _links_df["model_link_id"].tolist()
    WranglerLogger.debug(f"CONTROL - Number of total links: {len(all_model_link_ids)}")

    assert set(test_links_of_selection) == set(control_links_of_selection)
    WranglerLogger.info(f"--Finished: {request.node.name}")


@pytest.mark.xfail
def test_identify_segment_ends(request, stpaul_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    from network_wrangler.roadway.segment import identify_segment_endpoints

    # TODO FIXME
    net = stpaul_net

    _df = identify_segment_endpoints(net)

    calculated_d = _df.groupby("segment_id")["model_node_id"].apply(list).to_dict()
    correct_d = {
        0: [4799, 68320],
        1: [45635, 106816],
        2: [35752, 66416],
        3: [32501, 62153],
        4: [7734, 77606],
        5: [7715, 61777],
        6: [12163, 39514],
        7: [100805, 161661],
        8: [122991, 123042],
        9: [57530, 118107],
        10: [35752, 44233],
        11: [44190, 123002],
        12: [47478, 47522],
        13: [47478, 49757],
        14: [57601, 57637],
        15: [60956, 106816],
        16: [66415, 123049],
        17: [71838, 134504],
        18: [75313, 134918],
        19: [78880, 78885],
    }
    WranglerLogger.debug(f"Expected segment: {correct_d}")
    WranglerLogger.debug(f"Calculated segment: {calculated_d}")

    assert calculated_d == correct_d
    WranglerLogger.info(f"--Finished: {request.node.name}")


# selection, answer
query_tests = [
    # TEST 2
    (
        # SELECTION 2
        {"name": ["6th", "Sixth", "sixth"], "drive_access": 1},
        # ANSWER 2
        '((name.str.contains("6th") or '
        + 'name.str.contains("Sixth") or '
        + 'name.str.contains("sixth")) and '
        + "drive_access==1)",
    ),
    # TEST 2
    (
        # SELECTION 2
        {
            "name": [
                "6th",
                "Sixth",
                "sixth",
            ],  # find streets that have one of the various forms of 6th
            "lanes": [1, 2],  # only select links that are either 1 or 2 lanes
            "bike_access": [1],  # only select links that are marked for biking
        },
        # ANSWER 2
        '((name.str.contains("6th") or '
        + 'name.str.contains("Sixth") or '
        + 'name.str.contains("sixth")) and '
        + "(lanes==1 or lanes==2) and "
        + "(bike_access==1))",
    ),
    # TEST 3
    (
        # SELECTION 3
        {
            "model_link_id": [134574],
        },
        # ANSWER 3
        "((model_link_id==134574))",
    ),
]


@pytest.mark.parametrize("test_spec", query_tests)
def test_query_builder(request, test_spec):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    selection, answer = test_spec
    WranglerLogger.debug(f"Getting selection: \n{selection}")

    sel_query = dict_to_query(selection)

    WranglerLogger.debug(f"\nsel_query: \n{sel_query}")
    WranglerLogger.debug(f"\nanswer: \n{answer}")
    assert sel_query == answer

    WranglerLogger.info(f"--Finished: {request.node.name}")
