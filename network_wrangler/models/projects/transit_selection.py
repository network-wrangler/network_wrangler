"""Data Models for selecting transit trips, nodes, links, and routes."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Optional

from pydantic import ConfigDict, Field, field_validator

from .._base.records import RecordModel
from .._base.types import AnyOf, ForcedStr, OneOf, TimespanString, validate_timespan_string

SelectionRequire = Literal["any", "all"]


class SelectTripProperties(RecordModel):
    """Selection properties for transit trips."""

    trip_id: Annotated[list[ForcedStr] | None, Field(None, min_length=1)]
    shape_id: Annotated[list[ForcedStr] | None, Field(None, min_length=1)]
    direction_id: Annotated[int | None, Field(None)]
    service_id: Annotated[list[ForcedStr] | None, Field(None, min_length=1)]
    route_id: Annotated[list[ForcedStr] | None, Field(None, min_length=1)]
    trip_short_name: Annotated[list[ForcedStr] | None, Field(None, min_length=1)]

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        exclude_none=True,
        protected_namespaces=(),
    )


class TransitABNodesModel(RecordModel):
    """Single transit link model."""

    A: int | None = None  # model_node_id
    B: int | None = None  # model_node_id

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        exclude_none=True,
        protected_namespaces=(),
    )


class SelectTransitLinks(RecordModel):
    """Requirements for describing multiple transit links of a project card."""

    require_one_of: ClassVar[OneOf] = [
        ["ab_nodes", "model_link_id"],
    ]

    model_link_id: Annotated[list[int] | None, Field(min_length=1)] = None
    ab_nodes: Annotated[list[TransitABNodesModel] | None, Field(min_length=1)] = None
    require: SelectionRequire | None = "any"

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        exclude_none=True,
        protected_namespaces=(),
    )
    _examples: ClassVar[list[dict]] = [
        {
            "ab_nodes": [{"A": "75520", "B": "66380"}, {"A": "66380", "B": "75520"}],
            "type": "any",
        },
        {
            "model_link_id": [123, 321],
            "type": "all",
        },
    ]


class SelectTransitNodes(RecordModel):
    """Requirements for describing multiple transit nodes of a project card (e.g. to delete)."""

    require_any_of: ClassVar[AnyOf] = [
        [
            "model_node_id",
            # "stop_id_GTFS", TODO Not implemented
        ]
    ]

    # stop_id_GTFS: Annotated[Optional[List[ForcedStr]], Field(None, min_length=1)] TODO Not implemented
    model_node_id: Annotated[list[int], Field(min_length=1)]
    require: SelectionRequire | None = "any"

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        exclude_none=True,
        protected_namespaces=(),
    )

    _examples: ClassVar[list[dict]] = [
        # {"gtfstop_id": ["stop1", "stop2"], "require": "any"},  TODO Not implemented
        {"model_node_id": [1, 2], "require": "all"},
    ]


class SelectRouteProperties(RecordModel):
    """Selection properties for transit routes."""

    route_short_name: Annotated[list[ForcedStr] | None, Field(None, min_length=1)]
    route_long_name: Annotated[list[ForcedStr] | None, Field(None, min_length=1)]
    agency_id: Annotated[list[ForcedStr] | None, Field(None, min_length=1)]
    route_type: Annotated[list[int] | None, Field(None, min_length=1)]

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        exclude_none=True,
        protected_namespaces=(),
    )


class SelectTransitTrips(RecordModel):
    """Selection properties for transit trips."""

    trip_properties: SelectTripProperties | None = None
    route_properties: SelectRouteProperties | None = None
    timespans: Annotated[list[TimespanString] | None, Field(None, min_length=1)]
    nodes: SelectTransitNodes | None = None
    links: SelectTransitLinks | None = None

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        exclude_none=True,
        protected_namespaces=(),
    )

    @field_validator("timespans", mode="before")
    @classmethod
    def validate_timespans(cls, v):
        """Validate the timespans field."""
        if v is not None:
            return [validate_timespan_string(ts) for ts in v]
        return v
