"""Field types for GTFS data."""

from enum import IntEnum


class BikesAllowed(IntEnum):
    """Indicates whether bicycles are allowed."""

    NO_INFORMATION = 0
    ALLOWED = 1
    NOT_ALLOWED = 2


class DirectionID(IntEnum):
    """Indicates the direction of travel for a trip."""

    OUTBOUND = 0
    INBOUND = 1


class LocationType(IntEnum):
    """Indicates the type of node the stop record represents.

    Full documentation: https://gtfs.org/schedule/reference/#stopstxt
    """

    STOP_PLATFORM = 0
    STATION = 1
    ENTRANCE_EXIT = 2
    GENERIC_NODE = 3
    BOARDING_AREA = 4


class PickupDropoffType(IntEnum):
    """Indicates the pickup method for passengers at a stop.

    Full documentation: https://gtfs.org/schedule/reference
    """

    REGULAR = 0
    NONE = 1
    PHONE_AGENCY = 2
    COORDINATE_WITH_DRIVER = 3


class RouteType(IntEnum):
    """Indicates the type of transportation used on a route.

    Full documentation: https://gtfs.org/schedule/reference
    """

    TRAM = 0
    SUBWAY = 1
    RAIL = 2
    BUS = 3
    FERRY = 4
    CABLE_TRAM = 5
    AERIAL_LIFT = 6
    FUNICULAR = 7
    TROLLEYBUS = 11
    MONORAIL = 12


class TimepointType(IntEnum):
    """Indicates whether the specified time is exact or approximate.

    Full documentation: https://gtfs.org/schedule/reference
    """

    APPROXIMATE = 0
    EXACT = 1


class WheelchairAccessible(IntEnum):
    """Indicates whether the trip is wheelchair accessible.

    Full documentation: https://gtfs.org/schedule/reference
    """

    NO_INFORMATION = 0
    POSSIBLE = 1
    NOT_POSSIBLE = 2
