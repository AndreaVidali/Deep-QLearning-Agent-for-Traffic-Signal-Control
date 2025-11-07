import timeit

import numpy as np
import traci

from tlcs.generator import generate_routefile
from tlcs.utils import TestConfig

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class TestingSimulation:
    def __init__(
        self,
        model,
        sumo_cmd: list[str],
        config: TestConfig,
    ):
        self.model = model
        self.step = 0
        self.sumo_cmd = sumo_cmd
        self.max_steps = config.max_steps
        self.n_cars_generated = config.n_cars_generated
        self.green_duration = config.green_duration
        self.yellow_duration = config.yellow_duration
        self.num_states = config.num_states

        self.reward_episode: list[float] = []
        self.queue_length_episode: list[float] = []

    def run(self, episode):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        generate_routefile(
            seed=episode,
            max_steps=self.max_steps,
            n_cars_generated=self.n_cars_generated,
        )

        traci.start(self.sumo_cmd)
        print("Simulating...")

        # inits
        self.step = 0
        self.waiting_times = {}
        old_total_wait = 0
        old_action = -1  # dummy init

        while self.step < self.max_steps:
            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self.step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self.yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self.green_duration)

            # saving variables for later & accumulate reward
            old_action = action
            old_total_wait = current_total_wait

            self.reward_episode.append(reward)

        # print("Total reward:", np.sum(self.reward_episode))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time

    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo
        """
        # do not do more steps than the maximum allowed number of steps
        if (self.step + steps_todo) >= self.max_steps:
            steps_todo = self.max_steps - self.step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self.step += 1  # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self.queue_length_episode.append(queue_length)

    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            # get the road id where the car is located
            road_id = traci.vehicle.getRoadID(car_id)
            # consider only the waiting times of cars in incoming roads
            if road_id in incoming_roads:
                self.waiting_times[car_id] = wait_time
            else:
                # a car that was tracked has cleared the intersection
                if car_id in self.waiting_times:
                    del self.waiting_times[car_id]
        total_waiting_time = sum(self.waiting_times.values())
        return total_waiting_time

    def _choose_action(self, state):
        """
        Pick the best action known based on the current state of the env
        """
        return np.argmax(self.model.predict_one(state))

    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """

        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self.num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)

            # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road
            lane_pos = 750 - lane_pos

            # distance in meters from the traffic light -> mapping into cells
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
            elif lane_pos <= 750:
                lane_cell = 9

            # finding the lane where the car is located
            # x2TL_3 are the "turn left only" lanes
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                # composition of the two postion ID to create a number in interval 0-79
                car_position = int(str(lane_group) + str(lane_cell))
                valid_car = True

            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                # flag for not detecting cars crossing the intersection or driving away from it
                valid_car = False

            if valid_car:
                # write the position of the car car_id in the state array in the form of "cell occupied"
                state[car_position] = 1

        return state
