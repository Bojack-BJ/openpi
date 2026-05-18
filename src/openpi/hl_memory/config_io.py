from __future__ import annotations

import dataclasses
import pathlib
import sys
from typing import Any, get_args, get_origin


def load_yaml_mapping(path: pathlib.Path | str) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading `--config-yaml` requires `pyyaml` to be installed in the active environment."
        ) from exc

    path = pathlib.Path(path)
    payload = yaml.safe_load(path.read_text())
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a top-level mapping in {path}, got {type(payload).__name__}.")
    return dict(payload)


def build_dataclass_defaults(cls: type[Any], payload: dict[str, Any]) -> Any:
    field_map = {field.name: field for field in dataclasses.fields(cls)}
    unknown = sorted(key for key in payload if key not in field_map)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {', '.join(unknown)}")

    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        kwargs[key] = _coerce_value(value, field_map[key].type)
    return cls(**kwargs)


def resolve_cli_args_with_yaml(cls: type[Any], tyro_module: Any) -> Any:
    raw_args = sys.argv[1:]
    config_path: str | None = None
    forwarded_args: list[str] = []

    index = 0
    while index < len(raw_args):
        token = raw_args[index]
        if token in {"--config-yaml", "--config_yaml"}:
            if index + 1 >= len(raw_args):
                raise ValueError(f"{token} requires a file path.")
            config_path = raw_args[index + 1]
            index += 2
            continue
        forwarded_args.append(token)
        index += 1

    if config_path is None:
        return tyro_module.cli(cls, args=forwarded_args)

    defaults = build_dataclass_defaults(cls, load_yaml_mapping(config_path))
    return tyro_module.cli(cls, args=forwarded_args, default=defaults)


def _coerce_value(value: Any, annotation: Any) -> Any:
    if value is None:
        return None

    if annotation is pathlib.Path:
        return pathlib.Path(value)

    origin = get_origin(annotation)
    if origin is None:
        return value

    if origin in {list, tuple, set, frozenset, dict}:
        return value

    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1 and args[0] is pathlib.Path:
        return pathlib.Path(value)
    return value
