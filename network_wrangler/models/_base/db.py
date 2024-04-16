import copy
import hashlib

from collections import defaultdict
from typing import ClassVar, Callable

import pandas as pd
from pandera import DataFrameModel
from pandera.errors import SchemaErrors

from ...utils import fk_in_pk

from ...logger import WranglerLogger


class RequiredTableError(Exception):
    pass


class ForeignKeyValueError(Exception):
    pass


class DBModelMixin:
    """An mixin class for interrelated pandera DataFrameModel tables.

    Contains a bunch of convenience methods and overrides the dunder methods
        __deepcopy__ and __eq__.

    Methods:
        hash: hash of tables
        deepcopy: deepcopy of tables which references a custom __deepcopy__
        get_table: retrieve table by name
        table_names_with_field: returns tables in `table_names` with field name

    Attr:
        table_names: list of dataframe table names that are required as part of this "db"
            schema.
        optional_table_names: list of optional table names that will be added to `table_names` iff
            they are found.
        hash: creates a hash of tables found in `table_names` to track if they change.
        tables: dataframes corresponding to each table_name in `table_names`
        tables_dict: mapping of `<table_name>:<table>` dataframe
        _table_models: mapping of `<table_name>:<DataFrameModel>` to use for validation when
            `__setattr__` is called.
        _converters: mapping of `<table_name>:<converter_method>` where converter method should
            have a function signature of `(<table>, self.**kwargs)` .  Called on `__setattr__` if
            initial validation fails.

    Where metadata variable _fk = {<table_field>:[<fk table>,<fk field>]}

    e.g.: `_fk = {"parent_station": ["stops", "stop_id"]}`

    """

    # list of optional tables which are added to table_names if they are found.
    optional_table_names: ClassVar[list[str]] = []

    # list of interrelated tables.
    table_names: ClassVar[list[str]] = []

    # mapping of which Pandera DataFrameModel to validate the table to.
    _table_models: ClassVar[dict[str, DataFrameModel]] = {}

    # mapping of <table_name>:<conversion method> to use iff df validation fails.
    _converters: ClassVar[dict[str, Callable]] = {}

    def __setattr__(self, key, value):
        """
        Override the default setattr behavior to handle DataFrame validation.

        Note: this is NOT called when a dataframe is mutated in place!

        Args:
            key (str): The attribute name.
            value: The value to be assigned to the attribute.

        Raises:
            SchemaErrors: If the DataFrame does not conform to the schema.
            ForeignKeyError: If doesn't validate to foreign key.
        """
        if isinstance(value, pd.DataFrame):
            WranglerLogger.debug(f"Validating + coercing value to {key}")
            df = self.validate_coerce_table(key, value)
            super().__setattr__(key, df)
        else:
            super().__setattr__(key, value)

    def validate_coerce_table(
        self, table_name: str, table: pd.DataFrame
    ) -> pd.DataFrame:
        if table_name not in self._table_models:
            return table
        table_model = self._table_models[table_name]
        converter = self._converters.get(table_name)
        try:
            validated_df = table_model.validate(table, lazy=True)
        except SchemaErrors as e:
            if not converter:
                raise e
            WranglerLogger.debug(
                f"Initial validation failed as {table_name}. \
                                Attempting to convert using: {converter}"
            )
            # Note that some converters may have dependency on other attributes being set first
            converted_df = converter(table, **self.__dict__)
            validated_df = table_model.validate(converted_df, lazy=True)

        # Do this in both directions so that ordering of tables being added doesn't matter.
        self.check_table_fks(table_name, table=table)
        self.check_referenced_fks(table_name, table=table)
        return validated_df

    def initialize_tables(self, **kwargs):
        """
        Initializes the tables for the database.

        Args:
            **kwargs: Keyword arguments representing the tables to be initialized.

        Raises:
            RequiredTableError: If any required tables are missing in the initialization.
        """
        # Flag missing required tables
        _missing_tables = [t for t in self.table_names if t not in kwargs]
        if _missing_tables:
            raise RequiredTableError(
                f"{_missing_tables} is/are required \
                                        but not given in initialization."
            )

        # Add provided optional tables
        _opt_tables = [k for k in kwargs if k in self.optional_table_names]
        self.table_names += _opt_tables

        # Set tables in order
        for table in self.table_names:
            WranglerLogger.info(f"Initializing {table}")
            self.__setattr__(table, kwargs[table])

    @classmethod
    def fks(cls) -> list[dict]:
        """Return the fk field constraints as `{ <table>:{<field>:[<fk_table>,<fk_field>]} }`."""
        fk_fields = {}
        for table_name, table_model in cls._table_models.items():
            config = table_model.Config
            if not hasattr(config, "_fk"):
                continue
            fk_fields[table_name] = config._fk
        return fk_fields

    @classmethod
    def fields_as_fks(cls) -> list[str]:
        """Returns mapping of tables that have fields that other tables use as fks.

        `{ <table>:{<field>:[<table using FK>,<field using fk>]} }`

        Useful for knowing if you should check FK validation when changing a field value.
        """
        pks_as_fks = defaultdict(lambda: defaultdict(list))
        for t, field_fk in cls.fks().items():
            for f, fk in field_fk.items():
                fk_table, fk_field = fk
                pks_as_fks[fk_table][fk_field].append([t, f])
        return pks_as_fks

    def check_referenced_fk(
        self, table_name, field: str, table: pd.DataFrame = None
    ) -> bool:
        """True if table.field has the values referenced in any table referencing fields as fk.

        For example. If routes.route_id is referenced in trips table, we need to check that
        if a route_id is deleted, it isn't referenced in trips.route_id.
        """
        WranglerLogger.debug(
            f"Checking tables which referenced {table_name}.{field} as an FK"
        )
        if table is None:
            table = self.get_table(table_name)

        if field not in table:
            WranglerLogger.warning(
                f"Foreign key value {field} not in {table_name} \
                                    - skipping fk validation"
            )
            return True

        fields_as_fks = self.fields_as_fks()
        if table_name not in fields_as_fks:
            return True

        all_valid = True

        for ref_table_name, ref_field in fields_as_fks[table_name][field]:
            if ref_table_name not in self.table_names:
                WranglerLogger.debug(
                    f"Referencing table {ref_table_name} not in self.table_names \
                                    - skipping fk validation."
                )
                continue
            try:
                ref_table = self.get_table(ref_table_name)
            except RequiredTableError:
                WranglerLogger.warning(
                    f"Referencing table {ref_table_name} not yet set in \
                                       {type(self)} - skipping fk validation."
                )
                continue

            valid, _missing = fk_in_pk(table[field], ref_table[ref_field])
            all_valid = all_valid and valid
            if _missing:
                WranglerLogger.error(
                    f"Following values missing from {table_name}.{field} that \
                                    are referenced by {ref_table}:\n{_missing}"
                )
        return all_valid

    def check_referenced_fks(self, table_name: str, table: pd.DataFrame) -> bool:
        """True if this table has the values referenced in any table referencing fields as fk.

        For example. If routes.route_id is referenced in trips table, we need to check that
        if a route_id is deleted, it isn't referenced in trips.route_id.
        """
        WranglerLogger.debug(f"Checking referenced foreign keys for {table_name}")
        all_valid = True
        if table is None:
            table = self.get_table(table_name)
        all_valid = True
        for field in self.fields_as_fks()[table_name].keys():
            valid = self.check_referenced_fk(table_name, field, table=table)
            all_valid = valid and all_valid
        return all_valid

    def check_table_fks(
        self, table_name: str, table: pd.DataFrame, raise_error: bool = True
    ) -> bool:
        """Return True if the foreign key fields in table have valid references.

        Note: will return true and give a warning if the specified foreign key table does not exist.
        """
        WranglerLogger.debug(f"Checking foreign keys for {table_name}")
        fks = self.fks()
        if table_name not in fks:
            return True
        if table is None:
            table = self.get_table(table_name)
        all_valid = True
        for field, fk in fks[table_name].items():
            fk_table_name, fk_field = fk
            if field not in table:
                WranglerLogger.warning(
                    f"Foreign key value {field} not in {table_name} \
                    - skipping validation"
                )
                continue

            if fk_table_name not in self.table_names:
                WranglerLogger.warning(
                    f"Table {fk_table_name} for specified FK \
                                       {table_name}.{field} not in table list - skipping validation."
                )
                continue
            try:
                fk_table = self.get_table(fk_table_name)
            except RequiredTableError:
                WranglerLogger.warning(
                    f"Table {fk_table_name} for specified FK \
                                        {table_name}.{field} not in {type(self)} \
                                        - skipping validation."
                )
                continue
            if fk_field not in fk_table:
                WranglerLogger.error(
                    f"!!! {fk_table_name} missing {fk_field} field used as FK\
                                    ref in {table_name}.{field}."
                )
                all_valid = False
                continue
            valid, missing = fk_in_pk(table[field], fk_table[fk_field])
            if missing:
                WranglerLogger.error(
                    f"!!! {fk_table_name}.{fk_field} missing values used as FK\
                                    in {table_name}.{field}:\n_missing"
                )
            all_valid = valid and all_valid

        if not all_valid:
            if raise_error:
                raise ForeignKeyValueError(
                    "FK fields/ values referenced in {table_name} missing."
                )
            return False
        return True

    def check_fks(self) -> bool:
        """Check all FKs in set of tables."""
        all_valid = True
        for table_name in self.table_names:
            valid = self.check_table_fks(table_name, raise_error=False)
            all_valid = valid and all_valid
        return all_valid

    @property
    def tables(self) -> list[DataFrameModel]:
        return [self.__dict__[t] for t in self.table_names]

    @property
    def tables_dict(self) -> dict[str, DataFrameModel]:
        num_records = [len(self.get_table(t)) for t in self.table_names]
        return pd.DataFrame({"Table": self.table_names, "Records": num_records})

    @property
    def describe_df(self) -> pd.DataFrame:
        return pd.DataFrame({t: len(self.get_table(t)) for t in self.table_names})

    def get_table(self, table_name: str) -> pd.DataFrame:
        """Get table by name."""
        if table_name not in self.table_names:
            raise ValueError(f"{table_name} table not in db.")
        if table_name not in self.__dict__:
            raise RequiredTableError("Required table not set yet: {table_name}")
        return self.__dict__[table_name]

    def table_names_with_field(self, field: str) -> list[str]:
        """Returns tables in the class instance which contain the field."""
        return [t for t in self.table_names if field in self.get_table(t).columns]

    @property
    def hash(self) -> str:
        """A hash representing the contents of the tables in self.table_names"""
        _table_hashes = [self.get_table(t).df_hash() for t in self.table_names]
        _value = str.encode("-".join(_table_hashes))

        _hash = hashlib.sha256(_value).hexdigest()
        return _hash

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return self.hash == other.hash
        return False

    def __deepcopy__(self, memo):
        """Custom implementation of __deepcopy__ method.

        This method is called by copy.deepcopy() to create a deep copy of the object.

        Args:
            memo (dict): Dictionary to track objects already copied during deepcopy.

        Returns:
            Feed: A deep copy of the db object.
        """
        # Create a new, empty instance of the Feed class
        new_instance = self.__class__.__new__(self.__class__)

        # Copy all attributes to the new instance
        for attr_name, attr_value in self.__dict__.items():
            # Use copy.deepcopy to create deep copies of mutable objects
            if isinstance(attr_value, pd.DataFrame):
                setattr(new_instance, attr_name, copy.deepcopy(attr_value, memo))
            else:
                setattr(new_instance, attr_name, attr_value)

        WranglerLogger.warning(
            "Creating a deep copy of db object.\
            This will NOT update any referneces (e.g. from TransitNetwork)"
        )

        # Return the newly created deep copy instance of the object
        return new_instance

    def deepcopy(self):
        """Convenience method to exceute deep copy of instance."""
        return copy.deepcopy(self)
