from typing import Union, Tuple, Any
import hashlib
import re
from pathlib import Path
import pandas as pd

from ..logger import WranglerLogger


def topological_sort(adjacency_list, visited_list):
    """
    Topological sorting for Acyclic Directed Graph
    """

    output_stack = []

    def _topology_sort_util(vertex):
        if not visited_list[vertex]:
            visited_list[vertex] = True
            for neighbor in adjacency_list[vertex]:
                _topology_sort_util(neighbor)
            output_stack.insert(0, vertex)

    for vertex in visited_list:
        _topology_sort_util(vertex)

    return output_stack


def make_slug(text, delimiter: str = "_"):
    """
    makes a slug from text
    """

    text = re.sub("[,.;@#?!&$']+", "", text.lower())
    return re.sub("[\ ]+", delimiter, text)


def delete_keys_from_dict(dictionary: dict, keys: list) -> dict:
    """Removes list of keys from potentially nested dictionary.

    SOURCE: https://stackoverflow.com/questions/3405715/
    User: @mseifert

    Args:
        dictionary: dictionary to remove keys from
        keys: list of keys to remove

    """
    keys_set = set(keys)  # Just an optimization for the "if key in keys" lookup.

    modified_dict = {}
    for key, value in dictionary.items():
        if key not in keys_set:
            if isinstance(value, dict):
                modified_dict[key] = delete_keys_from_dict(value, keys_set)
            else:
                modified_dict[
                    key
                ] = value  # or copy.deepcopy(value) if a copy is desired for non-dicts.
    return modified_dict


def get_overlapping_range(ranges: list[Union[tuple[int], range]]) -> Union[None, range]:
    """Returns overlapping range for a list of ranges or tuples defining ranges.

    If no overlap found, returns None.
    """

    _ranges = [r if isinstance(r, range) else range(r[0], r[1]) for r in ranges]

    _overlap_start = max(r.start for r in _ranges)
    _overlap_end = min(r.stop for r in _ranges)

    if _overlap_start < _overlap_end:
        return range(_overlap_start, _overlap_end)
    else:
        return None


def coerce_val_to_series_type(val, s: pd.Series):
    """Coerces a calue to match type of pandas series.

    Will try not to fail so if you give it a value that can't convert to a number, it will
    return a string.

    Args:
        val: Any type of singleton value
        s (pd.Series): series to match the type to
    """
    # WranglerLogger.debug(f"Input val: {val} of type {type(val)} to match with series type \
    #    {pd.api.types.infer_dtype(s)}.")
    if pd.api.types.infer_dtype(s) in ["integer", "floating"]:
        try:
            v = float(val)
        except:
            v = str(val)
    elif pd.api.types.infer_dtype(s) == "boolean":
        v = bool(val)
    else:
        v = str(val)
    # WranglerLogger.debug(f"Return value: {v}")
    return v


def coerce_dict_to_df_types(
    d: dict, df: pd.DataFrame, skip_keys: list = [], return_skipped: bool = False
) -> dict:
    """Coerce dictionary values to match the type of a dataframe columns matching dict keys.

    Will also coerce a list of values.

    Args:
        d (dict): dictionary to coerce with singleton or list values
        df (pd.DataFrame): dataframe to get types from
        skip_keys: list of dict keys to skip. Defaults to []/
        return_skipped: keep the uncoerced, skipped keys/vals in the resulting dict.
            Defaults to False.

    Returns:
        dict: dict with coerced types
    """
    coerced_dict = {}
    for k, vals in d.items():
        if k in skip_keys:
            if return_skipped:
                coerced_dict[k] = vals
            continue
        if pd.api.types.infer_dtype(df[k]) == "integer":
            if isinstance(vals, list):
                coerced_v = [int(float(v)) for v in vals]
            else:
                coerced_v = int(float(vals))
        elif pd.api.types.infer_dtype(df[k]) == "floating":
            if isinstance(vals, list):
                coerced_v = [float(v) for v in vals]
            else:
                coerced_v = float(vals)
        elif pd.api.types.infer_dtype(df[k]) == "boolean":
            if isinstance(vals, list):
                coerced_v = [bool(v) for v in vals]
            else:
                coerced_v = bool(vals)
        else:
            if isinstance(vals, list):
                coerced_v = [str(v) for v in vals]
            else:
                coerced_v = str(vals)
        coerced_dict[k] = coerced_v
    return coerced_dict


def findkeys(node, kv):
    """Returns values of all keys in various objects.

    Adapted from arainchi on Stack Overflow:
    https://stackoverflow.com/questions/9807634/find-all-occurrences-of-a-key-in-nested-dictionaries-and-lists

    """
    if isinstance(node, list):
        for i in node:
            for x in findkeys(i, kv):
                yield x
    elif isinstance(node, dict):
        if kv in node:
            yield node[kv]
        for j in node.values():
            for x in findkeys(j, kv):
                yield x


def fk_in_pk(
    pk: Union[pd.Series, list], fk: Union[pd.Series, list]
) -> Tuple[bool, list]:
    if isinstance(fk, list):
        fk = pd.Series(fk)

    missing_flag = ~fk.isin(pk)

    if missing_flag.any():
        WranglerLogger.warning(
            f"Following keys referenced in {fk.name} but missing\
            in primary key table:\n{fk[missing_flag]} "
        )
        return False, fk[missing_flag].tolist()

    return True, []


def split_string_prefix_suffix_from_num(input_string: str):
    """Split a string prefix and suffix from *last* number.

    Define a regular expression pattern to capture:
    1. Any character sequence at the start, including digits (.*?), lazily.
    2. The last numeric sequence in the string (\d+), ensuring it's followed by
      non-digit characters (\D*) until the end of the string.
    """
    input_string = str(input_string)
    pattern = re.compile(r"(.*?)(\d+)(\D*)$")
    match = pattern.match(input_string)

    if match:
        # Extract the groups: prefix (including preceding numbers), last numeric part, suffix
        prefix, numeric_part, suffix = match.groups()
        # Convert the numeric part to an integer
        num_variable = int(numeric_part)
        return prefix, num_variable, suffix
    else:
        return input_string, 0, ""


def generate_new_id(
    input_id: str,
    existing_ids: pd.Series,
    id_scalar: int,
    iter_val: int = 10,
    max_iter: int = 1000,
) -> str:
    """Generate a new ID that isn't in existing_ids.

    TODO: check a registry rather than existing IDs

    args:
        input_id: id to use to generate new id.
        existing_ids: series that has existing IDs that should be unique
        id_scalar: scalar value to initially use to create the new id.
    """

    str_prefix, input_id, str_suffix = split_string_prefix_suffix_from_num(input_id)

    for i in range(1, max_iter + 1):
        new_id = f"{str_prefix}{int(input_id) + id_scalar + (iter_val * i)}{str_suffix}"
        if new_id not in existing_ids.values:
            return new_id
        elif i == max_iter:
            WranglerLogger.error(
                f"Cannot generate new id within max iters of {max_iter}."
            )
            raise ValueError("Cannot create unique new id.")


def generate_list_of_new_ids(
    input_ids: list[str],
    existing_ids: pd.Series,
    id_scalar: int,
    iter_val: int = 10,
    max_iter: int = 1000,
):
    # keep new_ids as list to preserve order
    new_ids = []
    existing_ids = set(existing_ids)
    for i in input_ids:
        new_id = generate_new_id(
            i,
            pd.Series(list(existing_ids)),
            id_scalar,
            iter_val=iter_val,
            max_iter=max_iter,
        )
        new_ids.append(new_id)
        existing_ids.add(new_id)
    return new_ids


def dict_to_hexkey(d: dict) -> str:
    """Converts a dictionary to a hexdigest of the sha1 hash of the dictionary.

    Args:
        d (dict): dictionary to convert to string

    Returns:
        str: hexdigest of the sha1 hash of dictionary
    """
    return hashlib.sha1(str(d).encode()).hexdigest()
