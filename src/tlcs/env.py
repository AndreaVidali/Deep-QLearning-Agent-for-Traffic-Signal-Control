from dataclasses import dataclass
from pathlib import Path

import numpy as np
import traci
from numpy.typing import NDArray
from sumolib import checkBinary

from tlcs.constants import (
    ACTION_TO_TL_PHASE,
    CELLS_PER_LANE_GROUP,
    INCOMING_EDGES,
    LANE_DISTANCE_TO_CELL,
    LANE_ID_TO_GROUP,
    ROAD_MAX_LENGTH,
    TL_GREEN_TO_YELLOW,
    TRAFFIC_LIGHT_ID,
)
from tlcs.generator import generate_routefile


@dataclass
class EnvStats:
    """Snapshot of environment statistics for a single simulation step."""

    queue_length: int


class Environment:
    """Reinforcement-learning environment wrapper around a SUMO traffic simulation."""

    def __init__(  # noqa: PLR0913
        self,
        state_size: int,
        n_cars_generated: int,
        max_steps: int,
        yellow_duration: int,
        green_duration: int,
        turn_chance: float,
        sumocfg_file: Path,
        gui: bool,
    ) -> None:
        """Initialize the environment.

        Args:
            state_size: Number of cells in the flattened state vector.
            n_cars_generated: Number of cars to generate for the episode.
            max_steps: Maximum number of simulation steps in an episode.
            yellow_duration: Number of steps to hold a yellow phase.
            green_duration: Number of steps to hold a green phase.
            turn_chance: Probability for each car to turn instead of going straight.
            sumocfg_file: Path to the SUMO configuration file.
            gui: Whether to use the SUMO GUI binary.
        """
        self.state_size = state_size
        self.n_cars_generated = n_cars_generated
        self.max_steps = max_steps
        self.yellow_duration = yellow_duration
        self.green_duration = green_duration
        self.turn_chance = turn_chance
        self.sumocfg_file = sumocfg_file
        self.gui = gui

        self.step = 0

    def build_sumo_cmd(self) -> list[str]:
        """Build the SUMO command line based on configuration settings.

        Returns:
            List of command-line arguments to start SUMO.
        """
        sumo_binary = checkBinary("sumo-gui" if self.gui else "sumo")

        if not self.sumocfg_file.exists():
            msg = f"SUMO config not found at '{self.sumocfg_file}'"
            raise FileNotFoundError(msg)

        return [
            sumo_binary,
            "-c",
            str(self.sumocfg_file),
            "--no-step-log",
            "true",
            "--waiting-time-memory",
            str(self.max_steps),
        ]

    def activate(self) -> None:
        """Start the SUMO simulation."""
        sumo_cmd = self.build_sumo_cmd()
        traci.start(sumo_cmd)

    def deactivate(self) -> None:
        """Stop the SUMO simulation."""
        traci.close()

    def is_over(self) -> bool:
        """Check whether the maximum number of steps has been reached.

        Returns:
            True if the episode is finished, False otherwise.
        """
        return self.step >= self.max_steps

    def generate_routefile(self, seed: int) -> None:
        """Generate a route file for the current episode.

        Args:
            seed: Random seed used for route generation.
        """
        generate_routefile(
            seed=seed,
            n_cars_generated=self.n_cars_generated,
            max_steps=self.max_steps,
            turn_chance=self.turn_chance,
        )

    def _get_lane_cell(self, lane_pos: float) -> int:
        """Map a continuous lane position to a discrete cell index.

        The lane is inverted so that 0 is at the traffic light and clamped to [0, ROAD_MAX_LENGTH].

        Args:
            lane_pos: Distance from the start of the edge in meters.

        Returns:
            Index of the discretized cell (0-based).
        """
        # invert so 0 is at the light; clamp to [0, ROAD_MAX_LENGTH]
        lane_pos = ROAD_MAX_LENGTH - lane_pos
        lane_pos = max(0.0, min(ROAD_MAX_LENGTH, lane_pos))

        for distance, cell in LANE_DISTANCE_TO_CELL.items():
            if lane_pos <= distance:
                return cell

        msg = "Error while getting lane cell."
        raise RuntimeError(msg)

    def get_state(self) -> NDArray:
        """Compute the discrete state representation of all vehicles.

        The state is a binary vector of length `state_size`. Each incoming lane is discretized into
        cells, grouped by direction and turning type.

        Returns:
            A NumPy array of shape (state_size,) with 0/1 occupancy values.
        """
        state = np.zeros(self.state_size, dtype=float)

        for car_id in traci.vehicle.getIDList():
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_group = LANE_ID_TO_GROUP.get(lane_id)
            if lane_group is None:
                # Ignore cars that are not on incoming lanes.
                continue

            lane_pos: float = traci.vehicle.getLanePosition(car_id)
            lane_cell = self._get_lane_cell(lane_pos)

            car_position = lane_group * CELLS_PER_LANE_GROUP + lane_cell

            if car_position < 0 or car_position >= self.state_size:
                msg = "Out of bounds car position."
                raise ValueError(msg)

            state[car_position] = 1.0

        return state

    def get_cumulated_waiting_time(self) -> float:
        """Compute the sum of waiting times for vehicles on incoming edges.

        Returns:
            Total accumulated waiting time of all vehicles on incoming edges.
        """
        waiting_times = 0.0

        for car_id in traci.vehicle.getIDList():
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id not in INCOMING_EDGES:
                continue
            wait_time = float(traci.vehicle.getAccumulatedWaitingTime(car_id))
            waiting_times += wait_time

        return waiting_times

    def _set_yellow_phase(self, green_phase_code: int) -> None:
        """Switch the traffic light to the yellow phase corresponding to a green phase.

        Args:
            green_phase_code: Code of the current green phase.
        """
        yellow_phase_code = TL_GREEN_TO_YELLOW[green_phase_code]
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, yellow_phase_code)

    def _set_green_phase(self, green_phase_code: int) -> None:
        """Switch the traffic light to the given green phase.

        Args:
            green_phase_code: Code of the green phase to activate.
        """
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, green_phase_code)

    def _simulate(self, duration: int) -> list[EnvStats]:
        """Advance the simulation for a given number of steps.

        The actual number of steps is capped so as not to exceed `max_steps`.

        Args:
            duration: Desired number of simulation steps.

        Returns:
            A list of EnvStats, one entry per simulation step.
        """
        stats: list[EnvStats] = []
        steps_todo = min(duration, self.max_steps - self.step)

        for _ in range(steps_todo):
            traci.simulationStep()
            self.step += 1
            queue_length = self.get_queue_length()
            stats.append(EnvStats(queue_length=queue_length))

        return stats

    def execute(self, action: int) -> list[EnvStats]:
        """Execute an action by changing the traffic light phase.

        If the requested phase differs from the current one, a yellow phase is inserted before
        switching to the new green phase.

        Args:
            action: Discrete action index mapped to a traffic light phase.

        Returns:
            A list of EnvStats collected during the applied phases.
        """
        next_green_phase = ACTION_TO_TL_PHASE[action]
        current_green_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT_ID)

        stats: list[EnvStats] = []

        if next_green_phase != current_green_phase:
            self._set_yellow_phase(current_green_phase)
            stats_yellow = self._simulate(self.yellow_duration)
            stats.extend(stats_yellow)

        if self.is_over():
            return stats

        self._set_green_phase(next_green_phase)
        stats_green = self._simulate(self.green_duration)
        stats.extend(stats_green)

        return stats

    def get_queue_length(self) -> int:
        """Return the number of stopped vehicles on all incoming edges.

        Returns:
            Total number of vehicles with speed 0 on incoming edges.
        """
        halt_n = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_s = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_e = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_w = traci.edge.getLastStepHaltingNumber("W2TL")
        return int(halt_n + halt_s + halt_e + halt_w)
