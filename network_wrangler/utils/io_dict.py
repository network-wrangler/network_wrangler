"""Utility functions for loading dictionaries from files."""

from pathlib import Path
from typing import Union

from .utils import merge_dicts


def _load_yaml(path: Path) -> dict:
    """Load yaml file at path."""
    import yaml

    with open(path, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)
    return data


def _load_json(path: Path) -> dict:
    """Load json file at path."""
    import json

    with open(path, "r") as json_file:
        data = json.load(json_file)
    return data


def _load_toml(path: Path) -> dict:
    """Load toml file at path."""
    import toml

    with open(path, "r", encoding="utf-8") as toml_file:
        data = toml.load(toml_file)
    return data


def load_dict(path: Path) -> dict:
    """Load a dictionary from a file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Specified dict file {path} not found.")

    if path.suffix.lower() == ".toml":
        return _load_toml(path)
    elif path.suffix.lower() == ".json":
        return _load_json(path)
    elif path.suffix.lower() == ".yaml" or path.suffix.lower() == ".yml":
        return _load_yaml(path)
    raise NotImplementedError(f"Filetype {path.suffix} not implemented.")


def load_merge_dict(path: Union[Path, list[Path]]) -> dict:
    """Load and merge multiple dictionaries from files."""
    if not isinstance(path, list):
        path = [path]
    data = load_dict(path[0])
    for path_item in path[1:]:
        merge_dicts(data, load_dict(path_item))
    return data
