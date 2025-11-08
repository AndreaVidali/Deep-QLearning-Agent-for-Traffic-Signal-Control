import random
import timeit

import numpy as np
import traci
from rich import print

from tlcs.generator import generate_routefile
from tlcs.memory import Memory
from tlcs.model import TrainModel
from tlcs.settings import TrainingSettings

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class TrainingSimulation:
    def __init__(
        self,
        model: TrainModel,
        memory: Memory,
        sumo_cmd: list[str],
        settings: TrainingSettings,
    ) -> None:
        self.model = model
        self.memory = memory
        self.sumo_cmd = sumo_cmd
        self.max_steps = settings.max_steps
        self.n_cars_generated = settings.n_cars_generated
        self.green_duration = settings.green_duration
        self.yellow_duration = settings.yellow_duration
        self.num_states = settings.num_states
        self.num_actions = settings.num_actions
        self.training_epochs = settings.training_epochs

        self.reward_store: list[float] = []
        self.cumulative_wait_store: list[float] = []
        self.avg_queue_length_store: list[float] = []

    def run(self, episode: int, epsilon: float) -> float:
        """
        Runs an episode of simulation, then starts a training session.
        Returns the wall-clock simulation time in seconds.
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        generate_routefile(
            seed=episode,
            n_cars_generated=self.n_cars_generated,
            max_steps=self.max_steps,
        )
        traci.start(self.sumo_cmd)

        print("Simulating...")

        self._reset_episode_vars()

        old_total_wait = 0.0
        old_state: np.ndarray | int = -1
        old_action = -1

        while self.step < self.max_steps:
            # get current state of the intersection
            current_state = self._get_state()

            # reward of previous action: change in cumulative waiting time
            current_total_wait = self._collect_waiting_times()
            reward = float(old_total_wait - current_total_wait)

            # store transition (s, a, r, s')
            if self.step != 0:
                self.memory.add_sample((old_state, old_action, reward, current_state))

            # epsilon-greedy action selection
            action = self._choose_action(current_state, epsilon)

            # if action changed, insert yellow phase
            if self.step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self.yellow_duration)

            # execute chosen green phase
            self._set_green_phase(action)
            self._simulate(self.green_duration)

            # update trackers
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # accumulate only negative rewards for clearer trend
            if reward < 0:
                self.sum_neg_reward += reward

        traci.close()

        self._save_episode_stats()
        print(f"Total reward: {self.sum_neg_reward} | Epsilon: {round(epsilon, 2)}")

        simulation_time = round(timeit.default_timer() - start_time, 1)
        return simulation_time

    def _reset_episode_vars(self):
        self.step = 0
        self.waiting_times = {}
        self.sum_neg_reward = 0.0
        self.sum_waiting_time = 0
        self.sum_queue_length = 0

    def _simulate(self, steps_todo: int) -> None:
        """Execute steps in SUMO while gathering statistics."""
        # cap to remaining steps
        if (self.step + steps_todo) >= self.max_steps:
            steps_todo = self.max_steps - self.step

        while steps_todo > 0:
            traci.simulationStep()
            self.step += 1
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self.sum_queue_length += queue_length

            # 1 car waiting for 1 step == 1 second of delay
            self.sum_waiting_time += queue_length

    def _collect_waiting_times(self) -> float:
        """Retrieve the waiting time of every car in the incoming roads."""
        incoming_roads = {"E2TL", "N2TL", "W2TL", "S2TL"}
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            wait_time = float(traci.vehicle.getAccumulatedWaitingTime(car_id))
            # get the road id where the car is located
            road_id = traci.vehicle.getRoadID(car_id)

            # consider only the waiting times of cars in incoming roads
            if road_id in incoming_roads:
                self.waiting_times[car_id] = wait_time
            else:
                # a car that was tracked has cleared the intersection
                self.waiting_times.pop(car_id, None)
        return float(sum(self.waiting_times.values()))

    def _choose_action(self, state: np.ndarray, epsilon: float) -> int:
        """Choose exploration vs exploitation according to epsilon-greedy policy."""
        capped_epsilon = max(0.0, min(1.0, epsilon))

        if random.random() < capped_epsilon:
            # explore
            return random.randint(0, self.num_actions - 1)

        # exploit
        return int(np.argmax(self.model.predict_one(state)))

    def _set_yellow_phase(self, old_action: int) -> None:
        """Activate the correct yellow light combination in SUMO."""
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number: int) -> None:
        """Activate the correct green light combination in SUMO."""
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def _get_queue_length(self) -> int:
        """Retrieve the number of cars with speed = 0 in every incoming lane."""
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        return int(halt_N + halt_S + halt_E + halt_W)

    def _get_state(self) -> np.ndarray:
        """Retrieve the state of the intersection from SUMO, as cell occupancy."""
        state = np.zeros(self.num_states, dtype=float)
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

            # invert so 0 is at the light; clamp to [0, 750]
            lane_pos = max(0.0, min(750.0, 750.0 - lane_pos))

            # distance buckets to 10 cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            else:
                lane_cell = 9

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

            car_position = lane_group * 10 + lane_cell  # 0..79

            if car_position < 0 or car_position >= self.num_states:
                msg = "Out of bounds car position."
                raise ValueError(msg)

            state[car_position] = 1.0

        return state

    def _save_episode_stats(self) -> None:
        """Save stats of the episode to plot graphs at the end of the session."""
        self.reward_store.append(float(self.sum_neg_reward))
        self.cumulative_wait_store.append(float(self.sum_waiting_time))
        self.avg_queue_length_store.append(self.sum_queue_length / float(self.max_steps))
