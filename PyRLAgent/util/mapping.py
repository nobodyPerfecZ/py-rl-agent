from enum import Enum
from typing import Any, Union


def get_value(mapping: dict[str, Any], name: Union[str, Any]) -> Any:
    """
    Returns the value of a mapping, given the key or the value.

    This method checks if the name (key or value) is inside the mapping and returns an exception if it does not fulfill.

    Args:
        mapping (dict[str, Any]):
            The mapping from a string argument (key) into the value

        name (str | Any):
            The key or the value

    Returns:
        Any:
            The value of name from mapping
    """
    if isinstance(name, str):
        name = mapping.get(name)
        if name is None:
            raise ValueError(f"Illegal name. The argument should be in {list(mapping.keys())}!")
    else:
        if name not in mapping.values():
            raise ValueError(f"Illegal name. The argument should be in {list(mapping.values())}!")
    return name


def get_values(mapping: dict[str, Any], names: list[Union[str, Any]]) -> list[Any]:
    """
    Returns the values of a mapping, given the keys or the values.

    This method checks if each name (key or value) is inside the mapping and returns an exception if it does not
    fulfill.

    Args:
        mapping (dict[str, Any]):
            The mapping from a string argument (key) into the value

        names (list[str | Any]):
            The keys or the values

    Returns:
        list[Any]:
            The values of each name from mapping
    """
    return [get_value(mapping, name) for name in names]


def get_values_enum(mapping: dict[Enum, Any], keys: Union[Enum, list[Enum]]) -> Union[Any, list[Any]]:
    # TODO: Add documentation
    if isinstance(keys, Enum):
        # Case: single key is given
        return mapping.get(keys)
    else:
        # Case: multiple keys are given
        return [mapping.get(key) for key in keys]
