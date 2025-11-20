from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import BaseModel, Field, NonNegativeInt, PositiveFloat, PositiveInt, model_validator


class TrainingSettings(BaseModel):
    """Configuration options for training the RL agent."""

    # simulation
    gui: bool
    total_episodes: PositiveInt
    max_steps: PositiveInt
    n_cars_generated: PositiveInt
    green_duration: PositiveInt
    yellow_duration: PositiveInt
    turn_chanche: Annotated[float, Field(ge=0, le=1)]

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
    gamma: Annotated[float, Field(ge=0, le=1)]

    # paths
    sumocfg_file: Path

    @model_validator(mode="after")
    def check_memory_bounds(self) -> Self:
        """Ensure that memory_size_min is strictly smaller than memory_size_max."""
        if self.memory_size_min >= self.memory_size_max:
            msg = (
                f"memory_size_min ({self.memory_size_min}) must be smaller than "
                f"memory_size_max ({self.memory_size_max})"
            )
            raise ValueError(msg)
        return self


class TestingSettings(BaseModel):
    """Configuration options for testing a trained RL agent."""

    # simulation
    gui: bool
    max_steps: PositiveInt
    n_cars_generated: PositiveInt
    episode_seed: int
    yellow_duration: PositiveInt
    green_duration: PositiveInt
    turn_chanche: Annotated[float, Field(ge=0, le=1)]

    # agent
    gamma: Annotated[float, Field(ge=0, le=1)]

    # paths
    sumocfg_file: Path


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.
    """
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        msg = f"Invalid YAML format in {path}; expected a mapping at the top level"
        raise TypeError(msg)

    return data


def load_training_settings(settings_file: Path) -> TrainingSettings:
    """Load and validate training settings from a YAML file.

    Args:
        settings_file: Path to the training settings YAML file.

    Returns:
        A validated TrainingSettings instance.
    """
    return TrainingSettings.model_validate(load_yaml(settings_file))


def load_testing_settings(settings_file: Path) -> TestingSettings:
    """Load and validate testing settings from a YAML file.

    Args:
        settings_file: Path to the testing settings YAML file.

    Returns:
        A validated TestingSettings instance.
    """
    return TestingSettings.model_validate(load_yaml(settings_file))
