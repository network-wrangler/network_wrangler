"""Data models for roadway changes."""

from __future__ import annotations

import itertools
from datetime import datetime
from typing import Any, ClassVar, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    RootModel,
    ValidationError,
    field_validator,
    model_validator,
    validate_call,
)

from ...errors import ScopeConflictError
from ...logger import WranglerLogger
from ...params import (
    DEFAULT_CATEGORY,
    DEFAULT_DELETE_MODES,
    DEFAULT_SEARCH_MODES,
    DEFAULT_TIMESPAN,
)
from ...utils.time import dt_overlaps, str_to_time_list
from .._base.records import RecordModel
from .._base.root import RootListMixin
from .._base.types import AnyOf, OneOf, TimespanString, validate_timespan_string
from .roadway_selection import SelectLinksDict, SelectNodesDict


class IndivScopedPropertySetItem(BaseModel):
    """Value for setting property value for a single time of day and category."""

    model_config = ConfigDict(extra="forbid", exclude_none=True)

    category: Optional[Union[str, int]] = DEFAULT_CATEGORY
    timespan: Optional[TimespanString] = DEFAULT_TIMESPAN
    set: Optional[Any] = None
    existing: Optional[Any] = None
    overwrite_conflicts: Optional[bool] = False
    change: Optional[Union[int, float]] = None
    _examples = [
        {"category": "hov3", "timespan": ["6:00", "9:00"], "set": 2.0},
        {"category": "hov2", "set": 2.0},
        {"timespan": ["12:00", "2:00"], "change": -1},
    ]

    @property
    def timespan_dt(self) -> list[list[datetime]]:
        """Convert timespan to list of datetime objects."""
        return str_to_time_list(self.timespan)

    @model_validator(mode="before")
    @classmethod
    def check_set_or_change(cls, data: dict):
        """Validate that each item has a set or change value."""
        if not isinstance(data, dict):
            return data
        if data.get("set") and data.get("change"):
            WranglerLogger.warning("Both set and change are set. Ignoring change.")
            data["change"] = None

        WranglerLogger.debug(f"Data: {data}")
        if data.get("set") is None and data.get("change") is None:
            msg = f"Require at least one of 'set' or'change' in IndivScopedPropertySetItem"
            WranglerLogger.debug(msg=f"   Found: {data}")
            raise ValueError(msg)
        return data

    @model_validator(mode="before")
    @classmethod
    def check_categories_or_timespans(cls, data: Any) -> Any:
        """Validate that each item has a category or timespan value."""
        if not isinstance(data, dict):
            return data
        require_any_of = ["category", "timespan"]
        if not any(attr in data for attr in require_any_of):
            msg = f"Require at least one of {require_any_of}"
            raise ValidationError(msg)
        return data

    @field_validator("timespan")
    @classmethod
    def validate_timespan(cls, v):
        """Validate the timespan field."""
        if v is not None:
            return validate_timespan_string(v)
        return v


class GroupedScopedPropertySetItem(BaseModel):
    """Value for setting property value for a single time of day and category."""

    model_config = ConfigDict(extra="forbid", exclude_none=True)

    category: Optional[Union[str, int]] = None
    timespan: Optional[TimespanString] = None
    categories: Optional[list[Any]] = []
    timespans: Optional[list[TimespanString]] = []
    set: Optional[Any] = None
    overwrite_conflicts: Optional[bool] = False
    existing: Optional[Any] = None
    change: Optional[Union[int, float]] = None
    _examples = [
        {"category": "hov3", "timespan": ["6:00", "9:00"], "set": 2.0},
        {"category": "hov2", "set": 2.0},
        {"timespan": ["12:00", "2:00"], "change": -1},
    ]

    @model_validator(mode="before")
    @classmethod
    def check_set_or_change(cls, data: dict):
        """Validate that each item has a set or change value."""
        if not isinstance(data, dict):
            return data
        if "set" in data and "change" in data:
            WranglerLogger.warning("Both set and change are set. Ignoring change.")
            data["change"] = None
        return data

    @model_validator(mode="before")
    @classmethod
    def check_categories_or_timespans(cls, data: Any) -> Any:
        """Validate that each item has a category or timespan value."""
        if not isinstance(data, dict):
            return data
        require_any_of = ["category", "timespan", "categories", "timespans"]
        if not any(attr in data for attr in require_any_of):
            msg = f"Require at least one of {require_any_of}"
            raise ValidationError(msg)
        return data

    @field_validator("timespan")
    @classmethod
    def validate_timespan(cls, v):
        """Validate the timespan field."""
        if v is not None:
            return validate_timespan_string(v)
        return v

    @field_validator("timespans", mode="before")
    @classmethod
    def validate_timespans(cls, v):
        """Validate the timespans field."""
        if v is not None:
            return [validate_timespan_string(ts) for ts in v]
        return v


def _grouped_to_indiv_list_of_scopedpropsetitem(
    scoped_prop_set_list: list[Union[GroupedScopedPropertySetItem, IndivScopedPropertySetItem]],
) -> list[IndivScopedPropertySetItem]:
    """Converts a list of ScopedPropertySetItem to a list of IndivScopedPropertySetItem.

    If neither category or categories exists, replace with DEFAULT_CATEGORY
    If neither timespan or timespans exists, replace with DEFAULT_TIMESPAN

    Because each ScopedPropertySetItem can have multiple categories and timespans, this function
    explodes the list of ScopedPropertySetItem into a list of IndivScopedPropertySetItem, where
    each IndivScopedPropertySetItem has a single category and timespan, consistent with how it is
    stored at the link-level variables.
    """
    indiv_items = []
    for i in scoped_prop_set_list:
        try:
            item = IndivScopedPropertySetItem(**i)
        except Exception:
            item = GroupedScopedPropertySetItem(**i)

        if isinstance(item, IndivScopedPropertySetItem):
            indiv_items.append(item)
            continue

        # Create full lists of all categories and timespans
        categories = item.categories if item.categories else [DEFAULT_CATEGORY]
        timespans = item.timespans if item.timespans else [DEFAULT_TIMESPAN]

        for c, t in itertools.product(categories, timespans):
            indiv_item = IndivScopedPropertySetItem(
                category=c, timespan=t, set=item.set, change=item.change
            )
            indiv_items.append(indiv_item)

    return indiv_items


class ScopedPropertySetList(RootListMixin, RootModel):
    """List of ScopedPropertySetItems used to evaluate and apply changes to roadway properties."""

    root: list[IndivScopedPropertySetItem]

    @model_validator(mode="before")
    @classmethod
    def check_set_or_change(cls, data: list):
        """Validate that each item has a set or change value."""
        data = _grouped_to_indiv_list_of_scopedpropsetitem(data)
        return data

    @model_validator(mode="after")
    def check_conflicting_scopes(self):
        """Check for conflicting scopes in the list of ScopedPropertySetItem."""
        conflicts = []
        for i in self:
            if i.timespan == DEFAULT_TIMESPAN:
                continue
            overlapping_ts_i = self.overlapping_timespans(i.timespan)
            for j in overlapping_ts_i:
                if j == i:
                    continue
                if j.category == i.category:
                    conflicts.append((i, j))
        if conflicts:
            msg = "Conflicting scopes in ScopedPropertySetList"
            WranglerLogger.error(msg + f"\n    Conflicts: {conflicts}")
            raise ScopeConflictError(msg)

        return self

    def overlapping_timespans(self, timespan: TimespanString) -> list[IndivScopedPropertySetItem]:
        """Return a list of items that overlap with the given timespan."""
        timespan_dt = str_to_time_list(timespan)
        return [i for i in self if dt_overlaps(i.timespan_dt, timespan_dt)]

    @property
    def change_items(self) -> list[IndivScopedPropertySetItem]:
        """Filter out items that do not have a change value."""
        WranglerLogger.debug(f"self.root[0]: {self.root[0]}")
        return [i for i in self if i.change is not None]

    @property
    def set_items(self):
        """Filter out items that do not have a set value."""
        return [i for i in self if i.set is not None]


class RoadPropertyChange(RecordModel):
    """Value for setting property value for a time of day and category."""

    model_config = ConfigDict(extra="forbid", exclude_none=True)

    existing: Optional[Any] = None
    change: Optional[Union[int, float]] = None
    set: Optional[Any] = None
    scoped: Optional[Union[None, ScopedPropertySetList]] = None
    overwrite_scoped: Optional[Literal["conflicting", "all", "error"]] = None
    existing_value_conflict: Optional[Literal["error", "warn", "skip"]] = None

    require_one_of: ClassVar[OneOf] = [
        ["change", "set"],
    ]

    _examples: ClassVar[list] = [
        {"set": 1},
        {"existing": 2, "change": -1},
        {
            "set": 0,
            "scoped": [
                {"timespan": ["6:00", "9:00"], "value": 2.0},
                {"timespan": ["9:00", "15:00"], "value": 4.0},
            ],
        },
        {
            "set": 0,
            "scoped": [
                {
                    "categories": ["hov3", "hov2"],
                    "timespan": ["6:00", "9:00"],
                    "value": 2.0,
                },
                {"category": "truck", "timespan": ["6:00", "9:00"], "value": 4.0},
            ],
        },
        {
            "set": 0,
            "scoped": [
                {"categories": ["hov3", "hov2"], "value": 2.0},
                {"category": "truck", "value": 4.0},
            ],
        },
    ]


class RoadwayDeletion(RecordModel):
    """Requirements for describing roadway deletion project card (e.g. to delete)."""

    require_any_of: ClassVar[AnyOf] = [["links", "nodes"]]
    model_config = ConfigDict(extra="forbid")

    links: Optional[SelectLinksDict] = None
    nodes: Optional[SelectNodesDict] = None
    clean_shapes: bool = False
    clean_nodes: bool = False

    @field_validator("links")
    @classmethod
    def set_to_all_modes(cls, links: Optional[SelectLinksDict] = None):
        """Set the search mode to 'any' if not specified explicitly."""
        if links is not None and links.modes == DEFAULT_SEARCH_MODES:
            links.modes = DEFAULT_DELETE_MODES
        return links
