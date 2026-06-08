"""Configuration module for network_wrangler."""

from pathlib import Path

from ..logger import WranglerLogger
from .scenario import ScenarioConfig
from .utils import _config_data_from_files
from .wrangler import DefaultConfig, WranglerConfig

ConfigInputTypes = dict | Path | list[Path] | WranglerConfig


def load_wrangler_config(data: ConfigInputTypes | None = None) -> WranglerConfig:
    """Load the WranglerConfiguration."""
    if isinstance(data, WranglerConfig):
        return data
    if data is None:
        return WranglerConfig()
    if isinstance(data, dict):
        return WranglerConfig(**data)
    if isinstance(data, Path) or (
        isinstance(data, list) and all(isinstance(d, Path) for d in data)
    ):
        return load_wrangler_config(_config_data_from_files(data))
    msg = "No valid configuration data found."
    WranglerLogger.error(msg + f"\n   Found: {data}.")
    raise ValueError(msg)


def load_scenario_config(
    data: ScenarioConfig | Path | list[Path] | dict | None = None,
) -> ScenarioConfig:
    """Load the WranglerConfiguration."""
    if isinstance(data, ScenarioConfig):
        return data
    if isinstance(data, dict):
        return ScenarioConfig(**data)

    combined_data = _config_data_from_files(data)
    if combined_data is None:
        msg = "No scenario configuration data found."
        WranglerLogger.error(msg + f"\n  Data: {data}.")
        raise ValueError(msg)

    if isinstance(data, list):
        ex_path = data[0]
    elif isinstance(data, Path):
        ex_path = data
    else:
        ex_path = Path.cwd()

    if ex_path.is_file():
        ex_path = ex_path.parent

    scenario_config = ScenarioConfig(**combined_data, base_path=ex_path)
    return scenario_config
