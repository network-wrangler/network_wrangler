import copy
import os

from pandera.errors import SchemaError

from projectcard import read_card
from network_wrangler import WranglerLogger

"""
Usage:  `pytest tests/test_roadway/test_changes/test_roadway_add_delete.py`
"""


def test_add_roadway_link_project_card(request, small_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")

    _project = "Add Newsy Link"
    _category = "roadway_addition"
    _links = [
        {
            "A": 1,
            "B": 5,
            "model_link_id": 1155,
            "roadway": "bus_guideway",
            "walk_access": 0,
            "drive_access": 0,
            "bus_only": 1,
            "rail_only": 1,
            "bike_access": 0,
            "lanes": 1,
            "name": "new busway link",
        },
        {
            "A": 5,
            "B": 1,
            "model_link_id": 5511,
            "roadway": "bus_guideway",
            "walk_access": 0,
            "drive_access": 0,
            "bus_only": 1,
            "rail_only": 1,
            "bike_access": 0,
            "lanes": 1,
            "name": "new busway link",
        },
    ]

    pc_dict = {
        "project": _project,
        _category: {
            "links": _links,
        },
    }

    net = copy.deepcopy(small_net)
    net = net.apply(pc_dict)

    _new_link_idxs = [i["model_link_id"] for i in _links]
    _expected_new_link_fks = [(i["A"], i["B"]) for i in _links]
    _new_links = net.links_df.loc[_new_link_idxs]

    WranglerLogger.debug(f"New Links: \n{_new_links}")
    assert len(_new_links) == len(_links)

    assert set(list(zip(_new_links.A, _new_links.B))) == set(_expected_new_link_fks)

    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_add_roadway_project_card(request, stpaul_net, stpaul_ex_dir):
    WranglerLogger.info(f"--Starting: {request.node.name}")

    net = copy.deepcopy(stpaul_net)
    card_name = "road.add.simple.yml"
    expected_net_links = 2
    expected_net_nodes = 0

    project_card_path = os.path.join(stpaul_ex_dir, "project_cards", card_name)
    project_card = read_card(project_card_path, validate=False)

    orig_links_count = len(net.links_df)
    orig_nodes_count = len(net.nodes_df)
    net = net.apply(project_card.__dict__)
    net_links = len(net.links_df) - orig_links_count
    net_nodes = len(net.nodes_df) - orig_nodes_count

    assert net_links == expected_net_links
    assert net_nodes == expected_net_nodes
    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_add_delete_roadway_project_card(request, stpaul_net, stpaul_ex_dir):
    WranglerLogger.info(f"--Starting: {request.node.name}")

    net = copy.deepcopy(stpaul_net)
    card_name = "road.add_and_delete.yml"
    expected_net_links = -2 + 2
    expected_net_nodes = +1 - 1 + 1

    project_card_path = os.path.join(stpaul_ex_dir, "project_cards", card_name)
    project_card = read_card(project_card_path, validate=False)

    orig_links_count = len(net.links_df)
    orig_nodes_count = len(net.nodes_df)
    net = net.apply(project_card)
    net_links = len(net.links_df) - orig_links_count
    net_nodes = len(net.nodes_df) - orig_nodes_count

    assert net_links == expected_net_links
    assert net_nodes == expected_net_nodes
    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_delete_roadway_shape(request, stpaul_net, stpaul_ex_dir):
    WranglerLogger.info(f"--Starting: {request.node.name}")

    net = copy.deepcopy(stpaul_net)

    card_name = "road.delete.simple.yml"
    project_card_path = os.path.join(stpaul_ex_dir, "project_cards", card_name)
    project_card = read_card(project_card_path, validate=False)

    expected_net_links = -1

    orig_links_count = len(net.links_df)

    net = net.apply(project_card)
    net_links = len(net.links_df) - orig_links_count

    assert net_links == expected_net_links

    print("--Finished:", request.node.name)


def test_add_nodes(request, small_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    net = copy.deepcopy(small_net)

    node_properties = {
        "X": -93.14412,
        "Y": 44.87497,
        "bike_node": 1,
        "drive_node": 1,
        "transit_node": 0,
        "walk_node": 1,
        "model_node_id": 1234567,
    }

    net = net.apply(
        {
            "project": "test adding a node",
            "roadway_addition": {"nodes": [node_properties]},
        }
    )

    WranglerLogger.debug(
        f"Added Node 1234567: \n{net.nodes_df.loc[net.nodes_df.model_node_id == 1234567]}"
    )

    assert 1234567 in net.nodes_df.model_node_id.tolist()
    assert 1234567 in net.nodes_df.index.tolist()

    # should fail when adding a node with a model_node_id that already exists
    bad_node_properties = node_properties.copy()
    bad_node_properties["model_node_id"] = 1
    WranglerLogger.debug("Trying to add node 3494 into network but should fail b/c already there")
    try:
        net = net.apply(
            {
                "project": "test adding a node already in network",
                "roadway_addition": {"nodes": [bad_node_properties]},
            },
        )
        WranglerLogger.error("Should have failed due to overlapping node IDs")
        assert False
    except SchemaError:
        "expected ValueError when adding a node with a model_node_id that already exists"
    WranglerLogger.info(f"--Finished: {request.node.name}")
