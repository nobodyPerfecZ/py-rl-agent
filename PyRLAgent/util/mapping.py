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
