import copy

import pandas as pd
import pytest
from projectcard import read_card

from network_wrangler import WranglerLogger


def test_change_roadway_existing_and_change_single_link(request, stpaul_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    net = copy.deepcopy(stpaul_net)

    # Set facility selection
    _facility = {
        "links": {"osm_link_id": ["223371529"]},
        "from": {"osm_node_id": "187854529"},  # Jackson St
        "to": {"osm_node_id": "187899923"},  # Robert St N
    }
    _properties = {
        "lanes": {  # changes number of lanes 3 to 2 (reduction of 1)
            "existing": 2,
            "change": -1,
        }
    }
    _project_card_dict = {
        "project": "test",
        "roadway_property_change": {
            "facility": _facility,
            "property_changes": _properties,
        },
    }

    _selected_link_idx = net.get_selection(_facility).selected_links
    _p_to_track = ["name", "projects", *list(_properties.keys())]

    WranglerLogger.debug(f"_p_to_track: {_p_to_track}")

    _orig_links = pd.DataFrame(copy.deepcopy(net.links_df))
    _orig_links = _orig_links.loc[_selected_link_idx, _p_to_track]
    WranglerLogger.debug(f"_orig_links: \n{_orig_links}")

    # apply change
    net = net.apply(_project_card_dict)

    _rev_links = pd.DataFrame(net.links_df)
    _rev_links = _rev_links.loc[_selected_link_idx, _p_to_track]
    WranglerLogger.debug(f"_rev_links: \n{_rev_links}")

    WranglerLogger.debug(f"ORIGINAL to REVISED Comparison\n {_orig_links.compare(_rev_links)}")

    for p_name, p in _properties.items():
        _expected_value = p["existing"] + p["change"]
        WranglerLogger.debug(f"Expected_value of {p_name}: {_expected_value}")
        assert _rev_links[p_name].eq(_expected_value).all()
    assert _rev_links["projects"].eq([f"{_project_card_dict['project']},"]).all()
    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_change_multiple_properties_multiple_links(request, stpaul_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    net = copy.deepcopy(stpaul_net)
    # Set facility selection
    _facility = {
        "links": {"name": ["6th", "Sixth", "sixth"]},
        "from": {"osm_node_id": "187899923"},  # Jackson St
        "to": {"osm_node_id": "187865924"},  # Robert St N
    }
    _properties = {
        "lanes": {
            "set": 2,
        },
        "bus_only": {
            "set": True,
        },
        "drive_access": {
            "set": False,
        },
    }
    _project_card_dict = {
        "project": "test",
        "roadway_property_change": {
            "facility": _facility,
            "property_changes": _properties,
        },
    }
    _selected_link_idx = net.get_selection(_facility).selected_links
    _p_to_track = ["name", "projects", *list(_properties.keys())]

    WranglerLogger.debug(f"_p_to_track: {_p_to_track}")

    _orig_links = net.links_df.copy()
    _orig_links = _orig_links.loc[_selected_link_idx, _p_to_track]
    WranglerLogger.debug(f"_orig_links: \n{_orig_links}")

    # apply change
    net = net.apply(_project_card_dict)

    _rev_links = pd.DataFrame(net.links_df)
    _rev_links = _rev_links.loc[_selected_link_idx, _p_to_track]
    WranglerLogger.debug(f"_rev_links: \n{_rev_links}")

    WranglerLogger.debug(f"ORIGINAL to REVISED Comparison\n {_orig_links.compare(_rev_links)}")
    _rev_links = _rev_links.reset_index(drop=True)
    for p_name, p in _properties.items():
        _expected_value = p["set"]
        WranglerLogger.debug(f"Expected_value of {p_name}: {_expected_value}")
        assert _rev_links[p_name].eq(_expected_value).all()

    # make sure it doen't add project multiple times.
    WranglerLogger.debug(f"_rev_links['projects']:\n{_rev_links['projects']}")
    assert _rev_links.at[0, "projects"] == f"{_project_card_dict['project']},"
    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_change_multiple_properties_multiple_links_existing_set(request, stpaul_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    net = copy.deepcopy(stpaul_net)
    # Set facility selection
    _facility = {
        "links": {"name": ["6th", "Sixth", "sixth"]},
        "from": {"osm_node_id": "187899923"},  # Jackson St
        "to": {"osm_node_id": "187865924"},  # Robert St N
    }
    _properties = {
        "lanes": {
            "existing": 1,
            "set": 2,
        },
        "bus_only": {
            "set": 1,
        },
        "drive_access": {
            "set": 0,
        },
    }
    _project_card_dict = {
        "project": "test",
        "roadway_property_change": {
            "facility": _facility,
            "property_changes": _properties,
        },
    }
    _selection = net.get_selection(_facility)
    _p_to_track = ["name", *list(_properties.keys())]

    WranglerLogger.debug(f"_p_to_track: {_p_to_track}")

    # Store the selected link indices before applying the project
    # since the project changes (especially drive_access=0) will affect routing
    _selected_link_indices = _selection.selected_links

    _orig_links = pd.DataFrame(copy.deepcopy(net.links_df))
    _orig_links = _orig_links.loc[_selected_link_indices, _p_to_track]
    WranglerLogger.debug(f"_orig_links: \n{_orig_links}")

    # apply change
    net = net.apply(_project_card_dict)

    _rev_links = pd.DataFrame(net.links_df)
    # Use the stored indices instead of trying to get selection again
    _rev_links = _rev_links.loc[_selected_link_indices, _p_to_track]
    WranglerLogger.debug(f"_rev_links: \n{_rev_links}")

    WranglerLogger.debug(f"ORIGINAL to REVISED Comparison\n {_orig_links.compare(_rev_links)}")

    for p_name, p in _properties.items():
        _expected_value = p["set"]
        WranglerLogger.debug(f"Expected_value of {p_name}: {_expected_value}")
        assert _rev_links[p_name].eq(_expected_value).all()


def test_add_adhoc_field(request, small_net):
    """Makes sure new fields can be added in the API and be saved and read in again."""
    WranglerLogger.info(f"--Starting: {request.node.name}")
    net = copy.deepcopy(small_net)
    net.links_df["my_ad_hoc_field"] = 22.5

    WranglerLogger.debug(f"Network with field...\n{net.links_df['my_ad_hoc_field'].iloc[0:5]}")

    assert net.links_df["my_ad_hoc_field"].iloc[0] == 22.5
    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_add_default_value(request, stpaul_net):
    """Makes sure we can add a new field with a default value."""
    WranglerLogger.info(f"--Starting: {request.node.name}")
    net = copy.deepcopy(stpaul_net)

    _float_val = 1.2345
    _int_val = 10
    _str_val = "default value"

    _adhoc_props = {
        "my_ad_hoc_field_float": {"set": _float_val},
        "my_ad_hoc_field_integer": {"set": _int_val},
        "my_ad_hoc_field_string": {"set": _str_val},
    }

    _project_card_dict = {
        "project": "6th Street Ad Hoc Fields",
        "roadway_property_change": {
            "facility": {"links": {"all": True, "modes": ["any"]}},
            "property_changes": _adhoc_props,
        },
    }
    net = net.apply(_project_card_dict)

    WranglerLogger.debug(f"{net.links_df['my_ad_hoc_field_float'].value_counts()}")
    WranglerLogger.debug(f"{net.links_df['my_ad_hoc_field_integer'].value_counts()}")
    WranglerLogger.debug(f"{net.links_df['my_ad_hoc_field_string'].value_counts()}")

    assert (
        net.links_df.loc[net.links_df["my_ad_hoc_field_float"] == _float_val].shape
        == net.links_df.shape
    )
    assert (
        net.links_df.loc[net.links_df["my_ad_hoc_field_integer"] == _int_val].shape
        == net.links_df.shape
    )
    assert (
        net.links_df.loc[net.links_df["my_ad_hoc_field_string"] == _str_val].shape
        == net.links_df.shape
    )

    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_add_adhoc_field_from_card(request, stpaul_net, stpaul_ex_dir):
    """New fields can be added from a project card and that they will be the right type."""
    WranglerLogger.info(f"--Starting: {request.node.name}")
    net = copy.deepcopy(stpaul_net)
    project_card_name = "road.prop_change.new_fields.yml"

    project_card_path = stpaul_ex_dir / "project_cards" / project_card_name
    project_card = read_card(project_card_path)

    selected_link_indices = net.get_selection(
        project_card.roadway_property_change["facility"]
    ).selected_links
    attributes_to_update = list(project_card.roadway_property_change["property_changes"].keys())

    net = net.apply(project_card)

    rev_links = net.links_df.loc[selected_link_indices, attributes_to_update]
    rev_types = [(a, net.links_df[a].dtypes) for a in attributes_to_update]

    WranglerLogger.debug(f"Revised Links: \n{rev_links}\nNew Property Types: \n{rev_types}")

    assert net.links_df.loc[selected_link_indices[0], "my_ad_hoc_field_float"] == 1.1
    assert net.links_df.loc[selected_link_indices[0], "my_ad_hoc_field_integer"] == 2
    assert net.links_df.loc[selected_link_indices[0], "my_ad_hoc_field_string"] == "three"
    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_change_node_xy(request, small_net):
    """Tests if X and Y property changes from a project card also update the node/link geometry."""
    WranglerLogger.info(f"--Starting: {request.node.name}")
    net = copy.deepcopy(small_net)

    _test_link = net.links_df.iloc[0]
    _test_link_idx = _test_link.model_link_id
    _test_node = net.nodes_df.loc[[_test_link.A]].iloc[0]
    _test_node_idx = _test_node.model_node_id

    WranglerLogger.debug(f"Node Index: {_test_node_idx}")
    WranglerLogger.debug(f"Link Index: {_test_link_idx}")
    WranglerLogger.info(
        f"Original Node (Index: {_test_node_idx}): \n{net.nodes_df.loc[_test_node_idx]}"
    )

    facility = {
        "nodes": {"model_node_id": [int(_test_node_idx)]},
    }
    WranglerLogger.debug(f"facility: {facility}")
    _expected_X = -1000
    _expected_Y = 1000000
    properties = {
        "X": {"set": _expected_X},
        "Y": {"set": _expected_Y},
    }

    _project_card_dict = {
        "project": "Update node geometry",
        "roadway_property_change": {
            "facility": facility,
            "property_changes": properties,
        },
    }
    net = net.apply(_project_card_dict)

    # Make sure geometry and XY were updated in node
    _updated_node = net.nodes_df.loc[_test_node_idx]
    WranglerLogger.info(
        f"Updated Node: \n{_updated_node[['model_node_id', 'X', 'Y', 'geometry']]}"
    )
    assert _updated_node.geometry.x == _expected_X
    assert _updated_node.geometry.y == _expected_Y
    assert _expected_X == _updated_node.X
    assert _expected_Y == _updated_node.Y

    # Make sure geometry also updated in link e
    _updated_link = net.links_df.loc[_test_link_idx]
    WranglerLogger.info(
        f"Updated Link Geometry for ({_updated_link.A}-->{_updated_link.B}): \n\
            {_updated_link[['geometry']].values}"
    )
    _first_point_in_link = _updated_link.geometry.coords[0]
    assert (_first_point_in_link[0], _first_point_in_link[1]) == (
        _expected_X,
        _expected_Y,
    )

    WranglerLogger.info(f"--Finished: {request.node.name}")
