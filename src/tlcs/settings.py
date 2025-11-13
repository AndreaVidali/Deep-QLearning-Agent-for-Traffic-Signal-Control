from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import BaseModel, Field, NonNegativeInt, PositiveFloat, PositiveInt, model_validator


class TrainingSettings(BaseModel):
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
    learning_rate: PositiveFloat
    training_epochs: PositiveInt

    # memory
    memory_size_min: NonNegativeInt
    memory_size_max: PositiveInt

    # agent
    state_size: PositiveInt
    num_actions: PositiveInt
    gamma: Annotated[float, Field(ge=0, le=1)]

    # paths
    sumocfg_file: Path

    @model_validator(mode="after")
    def check_memory_bounds(self) -> Self:
        if self.memory_size_min >= self.memory_size_max:
            msg = (
                f"memory_size_min ({self.memory_size_min}) must be smaller "
                f"than memory_size_max ({self.memory_size_max})"
            )
            raise ValueError(msg)
        return self


class TestingSettings(BaseModel):
    # simulation
    gui: bool
    max_steps: PositiveInt
    n_cars_generated: PositiveInt
    episode_seed: int
    yellow_duration: PositiveInt
    green_duration: PositiveInt

    # agent
    state_size: PositiveInt
    num_actions: PositiveInt
    gamma: Annotated[float, Field(ge=0, le=1)]

    # paths
    sumocfg_file: Path
    model_to_test: PositiveInt


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dict."""
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        msg = f"Invalid YAML format in {path}"
        raise TypeError(msg)
    return data


def load_training_settings(settings_file: Path) -> TrainingSettings:
    """Load and validate a YAML training settings file."""
    data = load_yaml(settings_file)
    return TrainingSettings.model_validate(data)


def load_testing_settings(settings_file: Path) -> TestingSettings:
    """Load and validate a YAML testing settings file."""
    data = load_yaml(settings_file)
    return TestingSettings.model_validate(data)
