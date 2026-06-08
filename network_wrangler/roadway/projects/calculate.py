"""Wrapper function for applying code to change roadway network."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...logger import WranglerLogger

if TYPE_CHECKING:
    from ..network import RoadwayNetwork


def apply_calculated_roadway(
    roadway_net: RoadwayNetwork,
    pycode: str,
) -> RoadwayNetwork:
    """Changes roadway network object by executing pycode.

    Args:
        roadway_net: network to manipulate
        pycode: python code which changes values in the roadway network object
    """
    WranglerLogger.debug("Applying calculated roadway project.")
    self = roadway_net
    exec(pycode)
    # Invalidate network state since exec'd code may have modified links_df or nodes_df
    roadway_net._mark_modified()

    return roadway_net
