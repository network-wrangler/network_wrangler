from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, model_validator

from ...logger import WranglerLogger
from ...utils.utils import check_one_or_one_superset_present
from .types import AnyOf, ConflictsWith, OneOf


class OneOfError(ValueError):
    pass


class ConflictsError(ValueError):
    pass


class AnyOfError(ValueError):
    pass


class RecordModel(BaseModel):
    """A pydantic model for records which adds validation for require_any_of and require_one_of.

    Attributes:
        model_config (ConfigDict): Configuration dictionary for the model.
        require_any_of (ClassVar[AnyOf]): Class variable specifying fields that require at least
            one of them to be present.
        require_one_of (ClassVar[OneOf]): Class variable specifying fields that require exactly
            one of them to be present.
        conflicts_with: ClassVar[ConflictsWith]: Class variable specifying list of pairs of fields
            that cannot be present together.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        exclude_none=True,
        protected_namespaces=(),
    )
    require_conflicts_with: ClassVar[ConflictsWith] = []
    require_any_of: ClassVar[AnyOf] = []
    require_one_of: ClassVar[OneOf] = []
    _examples: ClassVar[list[Any]] = []

    @staticmethod
    def _check_field_exists(all_of_fields: str | list[str], fields_present: list[str]) -> bool:
        if isinstance(all_of_fields, list):
            return all(f in fields_present for f in all_of_fields)
        return all_of_fields in fields_present

    @model_validator(mode="before")
    def check_conflicts(cls, values):
        if not cls.require_one_of:
            return values

        _fields_present = [k for k, v in values.items() if v is not None]
        for pair in cls.require_conflicts_with:
            conflicts = [
                pair for pair in cls.require_conflicts_with if set(pair).issubset(_fields_present)
            ]

            if conflicts:
                _err_str = ", ".join(pair)
                WranglerLogger.error(
                    f"{cls} cannot have both: {'or both of: '.join(_err_str)}. \
                    \nFields present: \n{_fields_present}."
                )
                msg = f"{cls} failed `conflicts_with` validation."
                raise ConflictsError(msg)
        return values

    @model_validator(mode="before")
    def check_one_of(cls, values):
        """Return True if exactly one of the fields in require_one_of is in values.

        Ignores presence of None values.

        If one of the fields in require_one_of is a list, then return True for that list item if
            ALL of the fields in the list are in values.
        """
        if not cls.require_one_of:
            return values

        _fields_present = [k for k, v in values.items() if v is not None]

        _not_passing = [
            i
            for i in cls.require_one_of
            if not check_one_or_one_superset_present(i, _fields_present)
        ]

        if _not_passing:
            _err_str = list(map(str, _not_passing))
            WranglerLogger.error(
                f"{cls} should have exactly one of: {'and one of: '.join(_err_str)}. \
                Fields present: \n{_fields_present}."
            )
            msg = f"{cls} failed `one_of` validation."
            raise OneOfError(msg)
        return values

    @classmethod
    def _check_each_any_of(cls, fields: list[str], values) -> bool:
        if any(cls._check_field_exists(field, values) for field in fields):
            return True
        WranglerLogger.error(f"{cls} should have at least one of {fields}.")
        WranglerLogger.error(f"Given values: \n{values}.")
        return False

    @model_validator(mode="before")
    def check_any_of(cls, values):
        """Validates that at least one of the specified fields in `require_any_of` exists in `values`.

        When require_any_of is a list of lists, this method will check that at least one of the
        fields in all of the inner lists is present in `values`.

        Ignores presence of None values.

        Args:
            values (dict): The dictionary of field values to validate.

        Returns:
            dict: The original `values` if validation passes.

        Raises:
            ValueError: If none of the `require_any_of` conditions are met.
        """
        if not cls.require_any_of:
            return values

        if not isinstance(cls.require_any_of[0], list):
            _require_any_of = [cls.require_any_of]
        else:
            _require_any_of = cls.require_any_of
        _values = {k: v for k, v in values.items() if v is not None}
        if all(cls._check_each_any_of(fields, _values) for fields in _require_any_of):
            return values
        msg = f"{cls.__name__} requires at least one of the following fields: \
                        {cls.require_any_of}. None were provided."
        raise AnyOfError(msg)

    @property
    def asdict(self) -> dict:
        """Model as a dictionary."""
        return self.model_dump(exclude_none=True, by_alias=True)

    @property
    def fields(self) -> list[str]:
        """All fields in the selection."""
        return list(self.asdict.keys())
