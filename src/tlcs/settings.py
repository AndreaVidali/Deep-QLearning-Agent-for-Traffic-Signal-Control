from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt


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
    sumocfg_file: Path


class TestingSettings(BaseModel):
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
    sumocfg_file: Path
    model_to_test: PositiveInt
