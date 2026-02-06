"""ModelTransit class and functions for managing consistency between roadway and transit networks.

NOTE: this is not thoroughly tested and may not be fully functional.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from ..roadway.network import RoadwayNetwork

if TYPE_CHECKING:
    from ..transit.network import TransitNetwork


class ModelTransit:
    """ModelTransit class for managing consistency between roadway and transit networks."""

    def __init__(
        self,
        transit_net: TransitNetwork,
        roadway_net: RoadwayNetwork,
        shift_transit_to_managed_lanes: bool = True,
    ):
        """ModelTransit class for managing consistency between roadway and transit networks."""
        self.transit_net = transit_net
        self.roadway_net = roadway_net
        self._roadway_net_version = None
        self._transit_feed_hash = None
        self._transit_shifted_to_ML = shift_transit_to_managed_lanes

    @property
    def model_roadway_net(self):
        """ModelRoadwayNetwork associated with this ModelTransit."""
        return self.roadway_net.model_net

    @property
    def consistent_nets(self) -> bool:
        """Indicate if roadway and transit networks have changed since self.m_feed updated."""
        return bool(
            self.roadway_net.modification_version == self._roadway_net_version
            and self.transit_net.feed_hash == self._transit_feed_hash
        )

    @property
    def m_feed(self):
        """TransitNetwork.feed with updates for consistency with associated ModelRoadwayNetwork."""
        if self.consistent_nets:
            return self._m_feed
        # NOTE: look at this
        # If networks have changed, update model transit and update reference version/hash
        self._roadway_net_version = self.roadway_net.modification_version
        self._transit_feed_hash = copy.deepcopy(self.transit_net.feed_hash)

        if not self._transit_shifted_to_ML:
            self._m_feed = copy.deepcopy(self.transit_net.feed)
            return self._m_feed
        return None
