from network_wrangler.roadway import load_roadway_from_dir, write_roadway
from network_wrangler.transit import load_transit, write_transit
from projectcard import read_cards


def roadway_io(stpaul_ex_dir, test_out_dir):
    net = load_roadway_from_dir(stpaul_ex_dir)
    write_roadway(net, test_out_dir, prefix="stpaul", file_format="geojson")


def test_roadway_io(benchmark, stpaul_ex_dir, test_out_dir):
    benchmark(roadway_io, stpaul_ex_dir, test_out_dir)


def roadway_property_change(stpaul_ex_dir):
    net = load_roadway_from_dir(stpaul_ex_dir)
    c = stpaul_ex_dir / "project_cards" / "road.prop_change.multiple.yml"
    p = read_cards(c)
    for p in p.values():
        net.apply(p)


def test_roadway_property_change(benchmark, stpaul_ex_dir):
    benchmark(roadway_property_change, stpaul_ex_dir)


def roadway_model_net(stpaul_ex_dir, test_out_dir):
    net = load_roadway_from_dir(stpaul_ex_dir)
    c = stpaul_ex_dir / "project_cards" / "road.managed_lanes.restricted_access.yml"
    card_dict = read_cards(c)
    for p in card_dict.values():
        net = net.apply(p)
    net.model_net.write(test_out_dir / "stpaul_managed_lane_model_net")


def test_roadway_managed_lane_model_net(benchmark, stpaul_ex_dir, test_out_dir):
    benchmark(roadway_model_net, stpaul_ex_dir, test_out_dir)


def apply_many_road_projects(stpaul_ex_dir):
    net = load_roadway_from_dir(stpaul_ex_dir)
    projects = ["road.divide_facility.yml", "road.prop_change.multiple.yml"]
    c = [stpaul_ex_dir / "project_cards" / p for p in projects]
    card_dict = read_cards(c)
    for p in card_dict.values():
        net.apply(p)


def test_roadway_many_projects_io(benchmark, stpaul_ex_dir):
    benchmark(apply_many_road_projects, stpaul_ex_dir)


def transit_io(stpaul_ex_dir, test_out_dir):
    net = load_transit(stpaul_ex_dir)
    write_transit(net, test_out_dir)


def test_transit_io(benchmark, stpaul_ex_dir, test_out_dir):
    benchmark(transit_io, stpaul_ex_dir, test_out_dir)


def apply_transit_property_change(stpaul_ex_dir):
    net = load_transit(stpaul_ex_dir)
    c = stpaul_ex_dir / "project_cards" / "transit.prop_change.route_time.yml"
    card_dict = read_cards(c)
    for p in card_dict.values():
        net.apply(p)


def test_transit_property_change(benchmark, stpaul_ex_dir):
    benchmark(apply_transit_property_change, stpaul_ex_dir)


def apply_multiple_transit_projects(stpaul_ex_dir):
    net = load_transit(stpaul_ex_dir)
    projects = ["transit.prop_change.route_time.yml", "transit.prop_change.multiple_trip_time.yml"]
    c = [stpaul_ex_dir / "project_cards" / p for p in projects]
    cards_dict = read_cards(c)
    for p in cards_dict.values():
        net.apply(p)


def test_multiple_transit_property_change(benchmark, stpaul_ex_dir):
    benchmark(apply_multiple_transit_projects, stpaul_ex_dir)


def apply_transit_routing_change(stpaul_ex_dir):
    road_net = load_roadway_from_dir(stpaul_ex_dir)
    net = load_transit(stpaul_ex_dir)
    net.road_net = road_net
    c = stpaul_ex_dir / "project_cards" / "transit.routing_change.yml"
    p = read_cards(c)
    for p in p.values():
        net.apply(p)


def test_transit_routing_change(benchmark, stpaul_ex_dir):
    benchmark(apply_transit_routing_change, stpaul_ex_dir)
