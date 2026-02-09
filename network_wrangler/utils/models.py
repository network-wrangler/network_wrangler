"""Helper functions for data models."""

import copy
from functools import wraps
from pathlib import Path
from typing import Optional, Union, _GenericAlias, get_args, get_origin, get_type_hints

import geopandas as gpd
import pandas as pd
import pandera as pa
from pandas import DataFrame
from pandera import DataFrameModel
from pandera.errors import SchemaError, SchemaErrors
from pandera.typing import DataFrame as PanderaDataFrame
from pydantic import BaseModel, ValidationError, validate_call
from pydantic._internal._model_construction import ModelMetaclass

from ..logger import WranglerLogger
from ..params import LAT_LON_CRS, SMALL_RECS
from .data import coerce_val_to_df_types


# Convert StringDtype columns to object dtype to avoid numpy.issubdtype compatibility issues
# in pandas 2.2+ with Python 3.11+
def _convert_string_dtype_to_object(df: DataFrame) -> DataFrame:
    """Convert StringDtype columns to object dtype for compatibility with numpy.issubdtype.

    This fixes compatibility issues with pandas 2.2+ StringDtype and numpy.issubdtype
    in Python 3.11+ when used with pandera validation.
    """
    df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].astype(object)
    return df


class DatamodelDataframeIncompatableError(Exception):
    """Raised when a data model and a dataframe are not compatable."""


class TableValidationError(Exception):
    """Raised when a table validation fails."""


def empty_df_from_datamodel(
    model: DataFrameModel, crs: int = LAT_LON_CRS
) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
    """Create an empty DataFrame or GeoDataFrame with the specified columns.

    Args:
        model (BaseModel): A pandera data model to create empty [Geo]DataFrame from.
        crs: if schema has geometry, will use this as the geometry's crs. Defaults to LAT_LONG_CRS
    Returns:
        An empty [Geo]DataFrame that validates to the specified model.
    """
    schema = model.to_schema()
    data: dict[str, list] = {col: [] for col in schema.columns}

    if "geometry" in data:
        return model(gpd.GeoDataFrame(data, crs=crs))

    return model(pd.DataFrame(data))


def default_from_datamodel(data_model: pa.DataFrameModel, field: str):
    """Returns default value from pandera data model for a given field name."""
    if field in data_model.__fields__ and hasattr(data_model.__fields__[field][1], "default"):
        return data_model.__fields__[field][1].default
    return None


def fill_df_with_defaults_from_model(df, model):
    """Fill a DataFrame with default values from a Pandera DataFrameModel.

    Args:
        df: DataFrame to fill with default values.
        model: Pandera DataFrameModel to get default values from.
    """
    for c in df.columns:
        default_value = default_from_datamodel(model, c)
        if default_value is None:
            df[c] = df[c].where(pd.notna(df[c]), None)
        else:
            df[c] = df[c].fillna(default_value)
    return df


@validate_call(config={"arbitrary_types_allowed": True})
def validate_df_to_model(
    df: DataFrame, model: type, output_file: Path = Path("validation_failure_cases.csv")
) -> DataFrame:
    """Wrapper to validate a DataFrame against a Pandera DataFrameModel with better logging.

    Also copies the attrs from the input DataFrame to the validated DataFrame.

    Args:
        df: DataFrame to validate.
        model: Pandera DataFrameModel to validate against.
        output_file: Optional file to write validation errors to. Defaults to
            validation_failure_cases.csv.
    """
    attrs = copy.deepcopy(df.attrs)
    err_msg = f"Validation to {model.__name__} failed."
    try:
        # Convert StringDtype columns to object dtype before validation to avoid
        # numpy.issubdtype compatibility issues with pandas 2.2+ and Python 3.11+
        df = _convert_string_dtype_to_object(df)
        model_df = model.validate(df, lazy=True)
        model_df = fill_df_with_defaults_from_model(model_df, model)
        model_df.attrs = attrs
        return model_df
    except (TypeError, ValueError) as e:
        WranglerLogger.error(f"Validation to {model.__name__} failed.\n{e}")
        raise TableValidationError(err_msg) from e
    except SchemaErrors as e:
        # Log the summary of errors
        WranglerLogger.error(
            f"Validation to {model.__name__} failed with {len(e.failure_cases)} \
            errors: \n{e.failure_cases}"
        )

        # If there are many errors, save them to a file
        if len(e.failure_cases) > SMALL_RECS:
            error_file = output_file
            e.failure_cases.to_csv(error_file)
            WranglerLogger.info(f"Detailed error cases written to {error_file}")
        else:
            # Otherwise log the errors directly
            WranglerLogger.error("Detailed failure cases:\n%s", e.failure_cases)
        raise TableValidationError(err_msg) from e
    except SchemaError as e:
        WranglerLogger.error(f"Validation to {model.__name__} failed with error: {e}")
        WranglerLogger.error(f"Failure Cases:\n{e.failure_cases}")
        raise TableValidationError(err_msg) from e


def identify_model(
    data: Union[pd.DataFrame, dict], models: list
) -> Union[DataFrameModel, BaseModel]:
    """Identify the model that the input data conforms to.

    Args:
        data (Union[pd.DataFrame, dict]): The input data to identify.
        models (list[DataFrameModel,BaseModel]): A list of models to validate the input
          data against.
    """
    for m in models:
        try:
            if isinstance(data, pd.DataFrame):
                validate_df_to_model(data, m)
            else:
                m(**data)
            return m
        except ValidationError:
            continue
        except SchemaError:
            continue

    WranglerLogger.error(
        f"The input data isn't consistant with any provided data model.\
                         \nInput data: {data}\
                         \nData Models: {models}"
    )
    msg = "The input data isn't consistant with any provided data model."
    raise TableValidationError(msg)


def extra_attributes_undefined_in_model(instance: BaseModel, model: BaseModel) -> list:
    """Find the extra attributes in a pydantic model that are not defined in the model."""
    defined_fields = model.model_fields
    all_attributes = list(instance.model_dump(exclude_none=True, by_alias=True).keys())
    extra_attributes = [a for a in all_attributes if a not in defined_fields]
    return extra_attributes


def submodel_fields_in_model(model: type, instance: Optional[BaseModel] = None) -> list:
    """Find the fields in a pydantic model that are submodels."""
    types = get_type_hints(model)
    model_type = (ModelMetaclass, BaseModel)
    submodels = [f for f in model.model_fields if isinstance(types.get(f), model_type)]
    if instance is not None:
        defined = list(instance.model_dump(exclude_none=True, by_alias=True).keys())
        return [f for f in submodels if f in defined]
    return submodels


def coerce_extra_fields_to_type_in_df(
    data: BaseModel, model: BaseModel, df: pd.DataFrame
) -> BaseModel:
    """Coerce extra fields in data that aren't specified in Pydantic model to the type in the df.

    Note: will not coerce lists of submodels, etc.

    Args:
        data (dict): The data to coerce.
        model (BaseModel): The Pydantic model to validate the data against.
        df (pd.DataFrame): The DataFrame to coerce the data to.
    """
    out_data = copy.deepcopy(data)

    # Coerce submodels
    for field in submodel_fields_in_model(model, data):
        out_data.__dict__[field] = coerce_extra_fields_to_type_in_df(
            data.__dict__[field], model.__annotations__[field], df
        )

    for field in extra_attributes_undefined_in_model(data, model):
        try:
            v = coerce_val_to_df_types(field, data.model_extra[field], df)
        except ValueError as err:
            raise DatamodelDataframeIncompatableError() from err
        out_data.model_extra[field] = v
    return out_data


def _is_type_from_type_hint(type_hint_value, type_to_check):
    def check_type_hint(value):
        if isinstance(value, _GenericAlias):
            try:
                if issubclass(value, type_to_check):
                    return True
            except TypeError:
                try:
                    if issubclass(value.__origin__, type_to_check):
                        return True
                except:
                    pass
        return False

    if get_origin(type_hint_value) is Union:
        args = get_args(type_hint_value)
        for arg in args:
            if check_type_hint(arg):
                return True
    elif check_type_hint(type_hint_value):
        return True

    return False


def validate_call_pyd(func):
    """Decorator to validate the function i/o using Pydantic models without Pandera."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        type_hints = get_type_hints(func)
        # Modify the type hints to replace pandera DataFrame models with pandas DataFrames
        modified_type_hints = {
            key: value
            for key, value in type_hints.items()
            if not _is_type_from_type_hint(value, PanderaDataFrame)
        }

        new_func = func
        new_func.__annotations__ = modified_type_hints
        validated_func = validate_call(new_func, config={"arbitrary_types_allowed": True})

        return validated_func(*args, **kwargs)

    return wrapper


def order_fields_from_data_model(df: pd.DataFrame, model: DataFrameModel) -> pd.DataFrame:
    """Order the fields in a DataFrame to match the order in a Pandera DataFrameModel.

    Will add any fields that are not in the model to the end of the DataFrame.
    Will not add any fields that are in the model but not in the DataFrame.

    Args:
        df: DataFrame to order.
        model: Pandera DataFrameModel to order the DataFrame to.
    """
    model_fields = list(model.__fields__.keys())
    df_model_fields = [f for f in model_fields if f in df.columns]
    df_additional_fields = [f for f in df.columns if f not in model_fields]
    return df[df_model_fields + df_additional_fields]
