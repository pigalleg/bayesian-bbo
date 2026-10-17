"""Shared project settings loaded from the package's config.yaml."""

from math import isfinite
from pathlib import Path
from typing import Any

import yaml

CONFIG_KEYS = {"normalize_y", "n_restarts_optimizer", "random_state", "candidates", "length_caps"}
CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the required project YAML settings."""
    if not path.exists():
        raise FileNotFoundError(f"Project configuration file not found: {path}")
    with path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    if not isinstance(raw_config, dict):
        raise ValueError("Configuration must be a YAML mapping")

    unknown_keys = set(raw_config) - CONFIG_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown_keys)}")
    missing_keys = CONFIG_KEYS - set(raw_config)
    if missing_keys:
        raise ValueError(f"Missing configuration keys: {sorted(missing_keys)}")
    config = raw_config
    if not isinstance(config["normalize_y"], bool):
        raise ValueError("normalize_y must be a boolean")
    for key in ("n_restarts_optimizer", "random_state", "candidates"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    caps = config["length_caps"]
    if not isinstance(caps, dict) or not caps:
        raise ValueError("length_caps must be a nonempty mapping of function IDs to positive finite values")
    for function_id, cap in caps.items():
        if isinstance(function_id, bool) or not isinstance(function_id, int) or function_id < 1:
            raise ValueError("length_caps keys must be positive integer function IDs")
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not isfinite(cap) or cap <= 0:
            raise ValueError(f"length_caps[{function_id}] must be a positive finite number")
    if not set(range(1, 9)) <= caps.keys():
        raise ValueError("length_caps must include Functions 1 through 8")
    return config


CONFIG = load_config()
