import copy

import pytest
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
    assert _new_links.at[_new_link_idxs[0], "projects"] == f"{_project},"
    assert set(zip(_new_links.A, _new_links.B, strict=False)) == set(_expected_new_link_fks)

    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_add_roadway_project_card(request, stpaul_net, stpaul_ex_dir):
    WranglerLogger.info(f"--Starting: {request.node.name}")

    net = copy.deepcopy(stpaul_net)
    card_name = "road.add.simple.yml"
    expected_net_links = 2
    expected_net_nodes = 0

    project_card_path = stpaul_ex_dir / "project_cards" / card_name
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
    project_card_path = stpaul_ex_dir / "project_cards" / card_name
    project_card = read_card(project_card_path)
    net = net.apply(project_card)

    ADD_LINK_IDS = [999998, 999997]
    NEW_NODE_IDS = [354388]
    DEL_LINK_IDS = [281, 477533]
    DEL_NODE_IDS = [314159]

    assert set(ADD_LINK_IDS).issubset(net.links_df.model_link_id.tolist())
    assert set(NEW_NODE_IDS).issubset(net.nodes_df.model_node_id.tolist())
    assert not set(DEL_LINK_IDS).intersection(net.links_df.model_link_id.tolist())
    assert not set(DEL_NODE_IDS).intersection(net.nodes_df.model_node_id.tolist())
    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_delete_roadway(request, stpaul_net, stpaul_ex_dir):
    WranglerLogger.info(f"--Starting: {request.node.name}")

    net = copy.deepcopy(stpaul_net)

    card_name = "road.delete.simple.yml"
    project_card_path = stpaul_ex_dir / "project_cards" / card_name
    project_card = read_card(project_card_path, validate=False)
    net = net.apply(project_card)

    assert 477533 not in net.links_df.model_link_id.tolist()

    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_add_nodes(request, small_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    from network_wrangler.roadway.nodes.create import NodeAddError

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
    with pytest.raises(NodeAddError):
        net = net.apply(
            {
                "project": "test adding a node already in network",
                "roadway_addition": {"nodes": [bad_node_properties]},
            },
        )
    WranglerLogger.info(f"--Finished: {request.node.name}")
