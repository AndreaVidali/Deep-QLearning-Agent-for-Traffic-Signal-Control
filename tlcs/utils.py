import os
import sys
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt
from sumolib import checkBinary


class TrainConfig(BaseModel):
    # simulation
    gui: bool
    total_episodes: PositiveInt
    max_steps: PositiveInt
    n_cars_generated: PositiveInt
    green_duration: PositiveInt
    yellow_duration: PositiveInt

    # model
    num_layers: PositiveInt
    width_layers: PositiveInt
    batch_size: PositiveInt
    learning_rate: Annotated[float, Field(gt=0)]
    training_epochs: PositiveInt

    # memory
    memory_size_min: NonNegativeInt
    memory_size_max: PositiveInt

    # agent
    num_states: PositiveInt
    num_actions: PositiveInt
    gamma: Annotated[float, Field(ge=0, le=1)]

    # paths
    models_path: Path
    sumocfg_file: Path


class TestConfig(BaseModel):
    # simulation
    gui: bool
    max_steps: PositiveInt
    n_cars_generated: PositiveInt
    episode_seed: int
    yellow_duration: PositiveInt
    green_duration: PositiveInt

    # agent
    num_states: PositiveInt
    num_actions: PositiveInt
    gamma: Annotated[float, Field(ge=0, le=1)]

    # paths
    models_path: Path
    sumocfg_file: Path
    model_to_test: PositiveInt


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML format in {path}")
    return data


def import_train_configuration(config_file: str | Path) -> TrainConfig:
    """Load and validate a flat YAML training configuration."""
    data = _load_yaml(Path(config_file))
    return TrainConfig.model_validate(data)


def import_test_configuration(config_file: str | Path) -> TestConfig:
    """Load and validate a flat YAML testing configuration."""
    data = _load_yaml(Path(config_file))
    return TestConfig.model_validate(data)


def set_sumo(gui: bool, sumocfg_file_name: str, max_steps: int) -> list[str]:
    """
    Configure the SUMO command-line based on GUI flag and config file name.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    sumo_binary = checkBinary("sumo-gui" if gui else "sumo")

    # Build the full path to the SUMO configuration
    sumocfg_path = os.path.join("intersection", sumocfg_file_name)
    if not os.path.exists(sumocfg_path):
        raise FileNotFoundError(f"SUMO config not found at '{sumocfg_path}'")

    # Command to run SUMO
    sumo_cmd = [
        sumo_binary,
        "-c",
        sumocfg_path,
        "--no-step-log",
        "true",
        "--waiting-time-memory",
        str(max_steps),
    ]
    return sumo_cmd


def set_test_path(models_path_name: str, model_n: int) -> tuple[str, str]:
    """
    Returns a model path that identifies the model number provided as argument
    and a newly created 'test' path.
    """
    model_folder_path = os.path.join(os.getcwd(), models_path_name, f"model_{model_n}", "")

    if os.path.isdir(model_folder_path):
        plot_path = os.path.join(model_folder_path, "test", "")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        return model_folder_path, plot_path
    else:
        sys.exit("The model number specified does not exist in the models folder")
