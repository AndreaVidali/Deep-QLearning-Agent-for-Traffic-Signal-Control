from pathlib import Path
from typing import Any

import yaml
from sumolib import checkBinary

from tlcs.settings import TestingSettings, TrainingSettings


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dict."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML format in {path}")
    return data


def load_training_settings(settings_file: Path) -> TrainingSettings:
    """Load and validate a YAML training settings file."""
    data = load_yaml(settings_file)
    return TrainingSettings.model_validate(data)


def load_testing_settings(settings_file: Path) -> TestingSettings:
    """Load and validate a YAML testing settings file."""
    data = load_yaml(settings_file)
    return TestingSettings.model_validate(data)


def set_sumo(gui: bool, sumocfg_file: Path, max_steps: int) -> list[str]:
    """
    Configure the SUMO command-line based on GUI flag and config file name.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    sumo_binary = checkBinary("sumo-gui" if gui else "sumo")

    # Build the full path to the SUMO configuration
    if not sumocfg_file.exists():
        raise FileNotFoundError(f"SUMO config not found at '{sumocfg_file}'")

    # Command to run SUMO
    sumo_cmd = [
        sumo_binary,
        "-c",
        str(sumocfg_file),
        "--no-step-log",
        "true",
        "--waiting-time-memory",
        str(max_steps),
    ]
    return sumo_cmd
