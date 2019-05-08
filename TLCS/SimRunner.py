import traci
import numpy as np
import random

# phase codes based on tlcs.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7

# HANDLE THE SIMULATION OF THE AGENT
class SimRunner:
    def __init__(self, sess, model, memory, traffic_gen, total_episodes, gamma, max_steps, green_duration, yellow_duration, sumoCmd):
        self._sess = sess
        self._model = model
        self._memory = memory
        self._traffic_gen = traffic_gen
        self._total_episodes = total_episodes
        self._gamma = gamma
        self._eps = 0  # controls the explorative/exploitative payoff, I choosed epsilon-greedy policy
        self._steps = 0
        self._waiting_times = {}
        self._sumoCmd = sumoCmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._sum_intersection_queue = 0
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_intersection_queue_store = []


    # THE MAIN FUCNTION WHERE THE SIMULATION HAPPENS
    def run(self, episode):
        # first, generate the route file for this simulation and set up sumo
        self._traffic_gen.generate_routefile(episode)
        traci.start(self._sumoCmd)

        # set the epsilon for this episode
        self._eps = 1.0 - (episode / self._total_episodes)

        # inits
        self._steps = 0
        tot_neg_reward = 0
        old_total_wait = 0
        self._waiting_times = {}
        self._sum_intersection_queue = 0

        while self._steps < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._get_waiting_times()
            reward = old_total_wait - current_total_wait

            # saving the data into the memory
            if self._steps != 0:
                self._memory.add_sample((old_state, old_action, reward, current_state))

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._steps != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait
            if reward < 0:
                tot_neg_reward += reward

        self._save_stats(tot_neg_reward)
        print("Total reward: {}, Eps: {}".format(tot_neg_reward, self._eps))
        traci.close()

    # HANDLE THE CORRECT NUMBER OF STEPS TO SIMULATE
    def _simulate(self, steps_todo):
        if (self._steps + steps_todo) >= self._max_steps:  # do not do more steps than the maximum number of steps
            steps_todo = self._max_steps - self._steps
        self._steps = self._steps + steps_todo  # update the step counter
        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._replay()  # training
            steps_todo -= 1
            intersection_queue = self._get_stats()
            self._sum_intersection_queue += intersection_queue

    # RETRIEVE THE WAITING TIME OF EVERY CAR IN THE INCOMING LANES
    def _get_waiting_times(self):
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        for veh_id in traci.vehicle.getIDList():
            wait_time_car = traci.vehicle.getAccumulatedWaitingTime(veh_id)
            road_id = traci.vehicle.getRoadID(veh_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[veh_id] = wait_time_car
            else:
                if veh_id in self._waiting_times:
                    del self._waiting_times[veh_id]  # the car isnt in incoming roads anymore, delete his waiting time
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    # DECIDE WHETER TO PERFORM AN EXPLORATIVE OR EXPLOITATIVE ACTION = EPSILON-GREEDY POLICY
    def _choose_action(self, state):
        if random.random() < self._eps:
            return random.randint(0, self._model.num_actions - 1) # random action
        else:
            return np.argmax(self._model.predict_one(state, self._sess)) # the best action given the current state

    # SET IN SUMO THE CORRECT YELLOW PHASE
    def _set_yellow_phase(self, old_action):
        yellow_phase = old_action * 2 + 1 # obtain the yellow phase code, based on the old action
        traci.trafficlight.setPhase("TL", yellow_phase)

    # SET IN SUMO A GREEN PHASE
    def _set_green_phase(self, action_number):
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    # RETRIEVE THE STATS OF THE SIMULATION FOR ONE SINGLE STEP
    def _get_stats(self):
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        intersection_queue = halt_N + halt_S + halt_E + halt_W
        return intersection_queue

    # RETRIEVE THE STATE OF THE INTERSECTION FROM SUMO
    def _get_state(self):
        state = np.zeros(self._model.num_states)

        for veh_id in traci.vehicle.getIDList():
            lane_pos = traci.vehicle.getLanePosition(veh_id)
            lane_id = traci.vehicle.getLaneID(veh_id)
            lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to TL, lane_pos = 0
            lane_group = -1  # just dummy initialization
            valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            # distance in meters from the TLS -> mapping into cells
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

            # finding the lane where the car is located - _3 are the "turn left only" lanes
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

            if lane_group >= 1 and lane_group <= 7:
                veh_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                veh_position = lane_cell
                valid_car = True

            if valid_car:
                state[veh_position] = 1  # write the position of the car veh_id in the state array

        return state

    # RETRIEVE A GROUP OF SAMPLES AND UPDATE THE Q-LEARNING EQUATION, THEN TRAIN
    def _replay(self):
        batch = self._memory.get_samples(self._model.batch_size)
        if len(batch) > 0:  # if there is at least 1 sample in the batch
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._model.predict_batch(states, self._sess)  # predict Q(state), for every sample
            q_s_a_d = self._model.predict_batch(next_states, self._sess)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._model.num_states))
            y = np.zeros((len(batch), self._model.num_actions))

            for i, b in enumerate(batch):
                state, action, reward, next_state = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._model.train_batch(self._sess, x, y)  # train the NN

    # SAVE THE STATS OF THE EPISODE TO PLOT THE GRAPHS AT THE END OF THE SESSION
    def _save_stats(self, tot_neg_reward):
            self._reward_store.append(tot_neg_reward)  # how much negative reward in this episode
            self._cumulative_wait_store.append(self._sum_intersection_queue)  # total number of seconds waited by cars in this episode
            self._avg_intersection_queue_store.append(self._sum_intersection_queue / self._max_steps)  # average number of queued cars per step, in this episode

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_intersection_queue_store(self):
        return self._avg_intersection_queue_store
