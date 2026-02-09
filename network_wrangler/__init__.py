"""Network Wrangler Package."""

__version__ = "1.0-beta.3"

import warnings

# Configure pandas 3.0+ to use object dtype for strings instead of StringDtype.
# This maintains compatibility with pandera until it fully supports StringDtype.
# This must be set before any pandas operations.
import pandas as pd

if hasattr(pd.options, "future") and hasattr(pd.options.future, "infer_string"):
    pd.options.future.infer_string = False

# Suppress the specific FutureWarning from geopandas
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*convert_dtype parameter is deprecated.*"
)

from .configs import load_wrangler_config
from .logger import WranglerLogger, setup_logging
from .roadway.io import load_roadway, load_roadway_from_dir, write_roadway
from .scenario import Scenario, create_scenario, load_scenario
from .transit.io import load_transit, write_transit
from .utils.df_accessors import *

__all__ = [
    "Scenario",
    "WranglerLogger",
    "create_scenario",
    "load_roadway",
    "load_roadway_from_dir",
    "load_scenario",
    "load_transit",
    "load_wrangler_config",
    "setup_logging",
    "write_roadway",
    "write_transit",
]


TARGET_ROADWAY_NETWORK_SCHEMA_VERSION = "1"
TARGET_TRANSIT_NETWORK_SCHEMA_VERSION = "1"
TARGET_PROJECT_CARD_SCHEMA_VERSION = "1"

MIN_ROADWAY_NETWORK_SCHEMA_VERSION = "0"
MIN_TRANSIT_NETWORK_SCHEMA_VERSION = "0"
MIN_PROJECT_CARD_SCHEMA_VERSION = "1"
