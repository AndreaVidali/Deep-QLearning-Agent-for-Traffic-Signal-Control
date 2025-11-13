from dataclasses import dataclass
from pathlib import Path

import numpy as np
import traci
from numpy.typing import NDArray
from sumolib import checkBinary

from tlcs.constants import (
    ACTION_TO_TL_PHASE,
    LANE_DISTANCE_TO_CELL,
    ROAD_MAX_LENGTH,
    TL_GREEN_TO_YELLOW,
)
from tlcs.generator import generate_routefile


@dataclass
class EnvStats:
    queue_length: int


class Environment:
    def __init__(
        self,
        state_size: int,
        n_cars_generated: int,
        max_steps: int,
        yellow_duration: int,
        green_duration: int,
        sumocfg_file: Path,
        gui: bool,
    ) -> None:
        self.state_size = state_size
        self.n_cars_generated = n_cars_generated
        self.max_steps = max_steps
        self.yellow_duration = yellow_duration
        self.green_duration = green_duration
        self.sumocfg_file = sumocfg_file
        self.gui = gui

        self.step = 0

    def build_sumo_cmd(self) -> list[str]:
        """
        Configure the SUMO command-line based on GUI flag and config file name.
        """
        sumo_binary = checkBinary("sumo-gui" if self.gui else "sumo")

        # Build the full path to the SUMO configuration
        if not self.sumocfg_file.exists():
            msg = f"SUMO config not found at '{self.sumocfg_file}'"
            raise FileNotFoundError(msg)

        # Command to run SUMO
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
        sumo_cmd = self.build_sumo_cmd()
        traci.start(sumo_cmd)

    def deactivate(self) -> None:
        traci.close()

    def is_over(self) -> bool:
        return self.step >= self.max_steps

    def generate_routefile(self, seed: int) -> None:
        generate_routefile(
            seed=seed,
            n_cars_generated=self.n_cars_generated,
            max_steps=self.max_steps,
        )

    def _get_lane_cell(self, lane_pos: float) -> int:
        # invert so 0 is at the light; clamp to [0, 750]
        lane_pos = ROAD_MAX_LENGTH - lane_pos

        for distance, cell in LANE_DISTANCE_TO_CELL.items():
            if lane_pos < distance:
                return cell

        msg = "Error while getting lane cell"
        raise RuntimeError(msg)

    def get_state(self) -> NDArray:
        state = np.zeros(self.state_size, dtype=float)
        lanes_groups = {
            "W": {"center/right": ("W2TL_0", "W2TL_1", "W2TL_2"), "left": ("W2TL_3")},
            "N": {"center/right": ("N2TL_0", "N2TL_1", "N2TL_2"), "left": ("N2TL_3")},
            "E": {"center/right": ("E2TL_0", "E2TL_1", "E2TL_2"), "left": ("E2TL_3")},
            "S": {"center/right": ("S2TL_0", "S2TL_1", "S2TL_2"), "left": ("S2TL_3")},
        }

        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            lane_pos = float(traci.vehicle.getLanePosition(car_id))
            lane_id = traci.vehicle.getLaneID(car_id)

            # map lane_id to group 0..7
            if lane_id in lanes_groups["W"]["center/right"]:
                lane_group = 0
            elif lane_id == lanes_groups["W"]["left"]:
                lane_group = 1
            elif lane_id in lanes_groups["N"]["center/right"]:
                lane_group = 2
            elif lane_id == lanes_groups["N"]["left"]:
                lane_group = 3
            elif lane_id in lanes_groups["E"]["center/right"]:
                lane_group = 4
            elif lane_id == lanes_groups["E"]["left"]:
                lane_group = 5
            elif lane_id in lanes_groups["S"]["center/right"]:
                lane_group = 6
            elif lane_id == lanes_groups["S"]["left"]:
                lane_group = 7
            else:
                continue  # ignore cars not in incoming lanes

            # distance buckets to 10 cells
            lane_cell = self._get_lane_cell(lane_pos)

            car_position = lane_group * 10 + lane_cell  # 0..79

            if car_position < 0 or car_position >= self.state_size:
                msg = "Out of bounds car position."
                raise ValueError(msg)

            state[car_position] = 1.0

        return state

    def get_cumulated_waiting_time(self) -> float:
        """Retrieve the waiting time of every car in the incoming roads."""
        incoming_roads = {"E2TL", "N2TL", "W2TL", "S2TL"}
        car_list = traci.vehicle.getIDList()
        waiting_times = 0.0

        for car_id in car_list:
            wait_time = float(traci.vehicle.getAccumulatedWaitingTime(car_id))
            # get the road id where the car is located
            road_id = traci.vehicle.getRoadID(car_id)

            # consider only the waiting times of cars in incoming roads
            if road_id in incoming_roads:
                waiting_times += wait_time

        return waiting_times

    def _set_yellow_phase(self, green_phase_code: int) -> None:
        """Activate the correct yellow light combination in SUMO."""
        yellow_phase_code = TL_GREEN_TO_YELLOW[green_phase_code]
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, green_phase_code: int) -> None:
        """Activate the correct green light combination in SUMO."""
        traci.trafficlight.setPhase("TL", green_phase_code)

    def _simulate(self, duration: int) -> list[EnvStats]:
        stats = []
        steps_todo = min(duration, self.max_steps - self.step)

        for _ in range(steps_todo):
            traci.simulationStep()
            self.step += 1
            queue_length = self.get_queue_length()
            stats.append(EnvStats(queue_length=queue_length))

        return stats

    def execute(self, action: int) -> list[EnvStats]:
        next_green_phase = ACTION_TO_TL_PHASE[action]
        current_green_phase = traci.trafficlight.getPhase("TL")

        stats = []

        if next_green_phase != current_green_phase:
            self._set_yellow_phase(current_green_phase)
            stats_yellow = self._simulate(self.yellow_duration)
            stats += stats_yellow

        if self.is_over():
            return stats

        self._set_green_phase(next_green_phase)
        stats_green = self._simulate(self.green_duration)
        stats += stats_green

        return stats

    def get_queue_length(self) -> int:
        """Retrieve the number of cars with speed = 0 in every incoming lane."""
        halt_n = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_s = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_e = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_w = traci.edge.getLastStepHaltingNumber("W2TL")
        return int(halt_n + halt_s + halt_e + halt_w)
