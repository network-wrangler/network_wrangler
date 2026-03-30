"""Parameters for Network Wrangler which should not be changed by the user.

Parameters that are here are used throughout the codebase and are stated here for easy reference.
Additional parameters that are more narrowly scoped are defined in the appropriate modules.

Changing these parameters may have unintended consequences and should only be done
by developers who understand the codebase.
"""

LAT_LON_CRS: int = 4326

DEFAULT_TIMESPAN: list[str] = ["00:00", "24:00"]

DEFAULT_CATEGORY: str = "any"

DEFAULT_SEARCH_MODES: list = ["drive"]
"""
This is a logical choice because you typically don't want a segment search from A--> F 
that includes b,c,d,e (walk nodes) and B,C,D,E (drive modes) to mix and match.
"""

DEFAULT_DELETE_MODES: list = ["any"]

MODES_TO_NETWORK_LINK_VARIABLES: dict[str, list[str]] = {
    "drive": ["drive_access"],
    "bus": ["bus_only", "drive_access"],
    "rail": ["rail_only"],
    "ferry": ["ferry_only"],
    "transit": ["bus_only", "rail_only", "ferry_only", "drive_access"],
    "walk": ["walk_access"],
    "bike": ["bike_access"],
}

SMALL_RECS: int = 5
"""Number of records to display in a dataframe summary."""

STRICT_MATCH_FIELDS = ["osm_link_id", "osm_node_id"]

FEET_PER_MILE = 5280.0
"""Feet to miles conversion constant."""

METERS_PER_KILOMETER = 1000.0
"""Meters to kilometers conversion constant."""
