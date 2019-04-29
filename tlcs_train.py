# created by Andrea Vidali
# info@andreavidali.com

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import random
import numpy as np
import math
import traci
from sumolib import checkBinary
import timeit
import matplotlib.pyplot as plt
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # to kill warning about tensorflow
import tensorflow as tf

# phase codes based on tlcs.net.xml
PHASE_NS_GREEN = 0 # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2 # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4 # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6 # action 3 code 11
PHASE_EWL_YELLOW = 7

# sumo things - we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# generation of routes of cars
def generate_routefile(seed, max_steps):
    np.random.seed(seed)  # make tests reproducible

    # initializations
    low_mode = False
    standard_mode = False
    NS_mode = False
    EW_mode = False

    if seed % 4 == 0: # low density
        n_cars_generated = 600
        low_mode = True
        print("Mode: low")
        traffic_mode = 1 # used for plotting
    elif seed % 4 == 1: # high density
        n_cars_generated = 6000
        standard_mode = True
        print("Mode: high")
        traffic_mode = 2
    elif seed % 4 == 2: # main source is north/south
        n_cars_generated = 3000
        NS_mode = True
        print("Mode: north-south main")
        traffic_mode = 3
    elif seed % 4 == 3:  # main source is east/west
        n_cars_generated = 3000
        EW_mode = True
        print("Mode: east-west main")
        traffic_mode = 4

    # the generation of cars is distributed according to a weibull distribution
    timings = np.random.weibull(2, n_cars_generated)
    timings = np.sort(timings)

    # reshape the distribution to fit the interval 0:max_steps
    car_gen_steps = []
    min_old = math.floor(timings[1])
    max_old = math.ceil(timings[-1])
    min_new = 0
    max_new = max_steps
    for value in timings:
        car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

    car_gen_steps = np.rint(car_gen_steps) # round every value to int -> effective steps when a car will be generated

    # produce the file for cars generation, one car per line
    with open("intersection/tlcs_train.rou.xml", "w") as routes:
        print("""<routes>
        <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

        <route id="W_N" edges="W2TL TL2N"/>
        <route id="W_E" edges="W2TL TL2E"/>
        <route id="W_S" edges="W2TL TL2S"/>
        <route id="N_W" edges="N2TL TL2W"/>
        <route id="N_E" edges="N2TL TL2E"/>
        <route id="N_S" edges="N2TL TL2S"/>
        <route id="E_W" edges="E2TL TL2W"/>
        <route id="E_N" edges="E2TL TL2N"/>
        <route id="E_S" edges="E2TL TL2S"/>
        <route id="S_W" edges="S2TL TL2W"/>
        <route id="S_N" edges="S2TL TL2N"/>
        <route id="S_E" edges="S2TL TL2E"/>""", file=routes)

        if standard_mode == True or low_mode == True:
            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.75:  # choose direction: straight or turn - 75% of times the car goes straight
                    route_straight = np.random.randint(1, 5) # choose a random source & destination
                    if route_straight == 1:
                        print('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 2:
                        print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 3:
                        print('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    else:
                        print('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                else: # car that turn -25% of the time the car turns
                    route_turn = np.random.randint(1, 9) # choose random source source & destination
                    if route_turn == 1:
                        print('    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 2:
                        print('    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 3:
                        print('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 4:
                        print('    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 5:
                        print('    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 6:
                        print('    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 7:
                        print('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 8:
                        print('    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)

        if NS_mode == True:
            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform() # car goes straight or turns
                source = np.random.uniform() # choose the source
                destination_straight = np.random.uniform() # destination if the car goes straight
                destination_turn = np.random.randint(1, 5) # destination if the car turns
                if straight_or_turn < 0.75:
                    if source < 0.90: # choose source: N S or E W
                        if destination_straight < 0.5: # choose destination
                            print('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        else:
                            print('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    else: # source: E W
                        if destination_straight < 0.5: # choose destination
                            print('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        else:
                            print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                else: # behavior: turn
                    if source < 0.90: # choose source: N S or E W
                        if destination_turn == 1: # choose destination
                            print('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        elif destination_turn == 2:
                            print('    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        elif destination_turn == 3:
                            print('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        elif destination_turn == 4:
                            print('    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    else: # source: E W
                        if destination_turn == 1: # choose destination
                            print('    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        elif destination_turn == 2:
                            print('    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        elif destination_turn == 3:
                            print('    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        elif destination_turn == 4:
                            print('    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)

        if EW_mode == True:
            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                source = np.random.uniform()
                destination_straight = np.random.uniform()
                destination_turn = np.random.randint(1, 5)
                if straight_or_turn < 0.75: # choose behavior: straight or turn
                    if source < 0.90: # choose source: N S or E W
                        if destination_straight < 0.5: # choose destination
                            print('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        else:
                            print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    else: # source: N S
                        if destination_straight < 0.5: # choose destination
                            print('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        else:
                            print('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                else: # behavior: turn
                    if source < 0.90: # choose source: N S or E W
                        if destination_turn == 1: # choose destination
                            print('    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        elif destination_turn == 2:
                            print('    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        elif destination_turn == 3:
                            print('    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        elif destination_turn == 4:
                            print('    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    else: # source: N S
                        if destination_turn == 1: # choose destination
                            print('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        elif destination_turn == 2:
                            print('    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        elif destination_turn == 3:
                            print('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        elif destination_turn == 4:
                            print('    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)

        print("</routes>", file=routes)

        return traffic_mode


class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size

        # define the placeholders
        self._states = None
        self._actions = None

        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None

        # now setup the model
        self._define_model()

    # DEFINE THE STRUCTURE OF THE NEURAL NETWORK
    def _define_model(self):
        # placeholders
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)

        # list of nn layers
        fc1 = tf.layers.dense(self._states, 400, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 400, activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc2, 400, activation=tf.nn.relu)
        fc4 = tf.layers.dense(fc3, 400, activation=tf.nn.relu)
        fc5 = tf.layers.dense(fc4, 400, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc5, self._num_actions)

        # parameters
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

    # RETURNS THE OUTPUT OF THE NETWORK GIVEN A SINGLE STATE
    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states: state.reshape(1, self.num_states)})

    # RETURNS THE OUTPUT OF THE NETWORK GIVEN A BATCH OF STATES
    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    # TRAIN THE NETWORK
    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def var_init(self):
        return self._var_init

# HANDLE THE MEMORY
class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory # size of memory
        self._samples = []

    # ADD A SAMPLE INTO THE MEMORY
    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0) # if the length is greater than the size of memory, remove the oldest element

    # GET n_samples SAMPLES RANDOMLY FROM THE MEMORY
    def get_samples(self, n_samples):
        if n_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples)) # get all the samples
        else:
            return random.sample(self._samples, n_samples) # get "batch size" number of samples

# HANDLE THE SIMULATION OF THE AGENT
class SimRunner:
    def __init__(self, sess, model, memory, total_episodes, gamma, max_steps, green_duration, yellow_duration, sumoCmd):
        self._sess = sess
        self._model = model
        self._memory = memory
        self._total_episodes = total_episodes
        self._gamma = gamma
        self._eps = 0 # controls the explorative/exploitative payoff
        self._steps = 0
        self._waiting_times = {}
        self._sumoCmd = sumoCmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._sum_intersection_queue = 0

        self._reward_store_LOW = []
        self._cumulative_wait_store_LOW = []
        self._avg_intersection_queue_store_LOW = []

        self._reward_store_HIGH = []
        self._cumulative_wait_store_HIGH = []
        self._avg_intersection_queue_store_HIGH = []

        self._reward_store_NS = []
        self._cumulative_wait_store_NS = []
        self._avg_intersection_queue_store_NS = []

        self._reward_store_EW = []
        self._cumulative_wait_store_EW = []
        self._avg_intersection_queue_store_EW = []

    # THE MAIN FUCNTION WHERE THE SIMULATION HAPPENS
    def run(self, episode):
        # first, generate the route file for this simulation and set up sumo
        traffic_mode = generate_routefile(episode, self._max_steps)
        traci.start(self._sumoCmd)

        # set the epsilon for this episode
        self._eps = 1.0 - (episode / self._total_episodes)

        # inits
        self._steps = 0
        tot_neg_reward = 0
        old_total_wait = 0
        self._waiting_times = {}
        self._sum_intersection_queue = 0

        # simulation (self._steps updated in function "_simulate")
        while self._steps < self._max_steps:
            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative reward between actions)
            # waiting time = seconds waited since the car spawn, cumulated for every car in incoming lanes
            current_total_wait = self._get_waiting_times()
            reward = old_total_wait - current_total_wait

            # data saving into memory & training - if the sim is just started, there is no old_state
            if self._steps != 0:
                self._memory.add_sample((old_state, old_action, reward, current_state))

            # choose the action to perform based on the current state
            action = self._choose_action(current_state)

            # if the chosen action is different from the last one, its time for the yellow phase
            if self._steps != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the action selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving the variables for the next step & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait
            if reward < 0:
                tot_neg_reward += reward

        self._save_stats(traffic_mode, tot_neg_reward)
        print("Total reward: {}, Eps: {}".format(tot_neg_reward, self._eps))
        traci.close()

    # HANDLE THE CORRECT NUMBER OF STEPS TO SIMULATE
    def _simulate(self, steps_todo):
        if (self._steps + steps_todo) >= self._max_steps: # do not do more steps than the maximum number of steps
            steps_todo = self._max_steps - self._steps
        self._steps = self._steps + steps_todo # update the step counter
        while steps_todo > 0:
            traci.simulationStep() # simulate 1 step in sumo
            self._replay()  # training
            steps_todo -= 1
            intersection_queue = self._get_stats()
            self._sum_intersection_queue += intersection_queue

    # RETRIEVE THE WAITING TIME OF EVERY CAR IN THE INCOMING LANES
    def _get_waiting_times(self):
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        for veh_id in traci.vehicle.getIDList():
            wait_time_car = traci.vehicle.getAccumulatedWaitingTime(veh_id) # get the waiting time
            road_id = traci.vehicle.getRoadID(veh_id) # get the road id where the car is
            if road_id in incoming_roads: # consider only the waiting times of cars in incoming roads
                self._waiting_times[veh_id] = wait_time_car
            else: # the car isnt into an incoming road anymore, delete the corresponding waiting time
                if veh_id in self._waiting_times:
                    del self._waiting_times[veh_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    # DECIDE WHETER TO PERFORM AN EXPLORATIVE OR EXPLOITATIVE ACTION
    def _choose_action(self, state):
        if random.random() < self._eps: # epsilon controls the randomness of the action
            return random.randint(0, self._model.num_actions - 1) # random action
        else:
            return np.argmax(self._model.predict_one(state, self._sess)) # the best action given the current state (prediction from nn)

    # SET IN SUMO A YELLOW PHASE
    def _set_yellow_phase(self, old_action):
        yellow_phase = old_action * 2 + 1 # obtain the correct yellow phase number based on the old action
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
            lane_pos = 750 - lane_pos # inversion of lane so if it is close to TL, lane_pos = 0
            lane_group = -1 # just dummy initialization
            valid_car = False # flag for not detecting cars crossing the intersection or driving away from it

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

            # in which lane is the car? _3 are the "turn left only" lanes
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
                veh_position = int(str(lane_group) + str(lane_cell)) # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                veh_position = lane_cell
                valid_car = True

            if valid_car:
                state[veh_position] = 1 # write the position of the car veh_id in the state array

        return state

    # RETRIEVE A GROUP OF SAMPLES AND UPDATE THE Q-LEARNING EQUATION, THEN TRAIN
    def _replay(self):
        batch = self._memory.get_samples(self._model.batch_size) # retrieve a group of samples
        if len(batch) > 0:  # if there is at least 1 sample in the memory
            states = np.array([val[0] for val in batch]) # isolate the old states from the batch
            next_states = np.array([val[3] for val in batch]) # isolate the next states from the batch
            q_s_a = self._model.predict_batch(states, self._sess) # # predict Q-values starting from the old states
            q_s_a_d = self._model.predict_batch(next_states, self._sess) # predict Q-values starting from the next states

            # setup training arrays
            x = np.zeros((len(batch), self._model.num_states)) # x: batch_size X 80
            y = np.zeros((len(batch), self._model.num_actions)) # y: batch_size X 4
            for i, b in enumerate(batch):
                state, action, reward, next_state = b[0], b[1], b[2], b[3] # extract data from one sample
                current_q = q_s_a[i] # get the Q values predicted before from state of this sample
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i]) # update the Q value just for the action in the sample
                # x is the input, y is the output of NN
                x[i] = state
                y[i] = current_q # Q-values that include the updated Q-value

            self._model.train_batch(self._sess, x, y) # train the NN

    # SAVE THE STATS OF THE EPISODE TO PLOT THE GRAPHS AT THE END OF THE SESSION
    def _save_stats(self, traffic_mode, tot_neg_reward):
        if traffic_mode == 1: # data low
            self._reward_store_LOW.append(tot_neg_reward) # how much negative reward in this episode
            self._cumulative_wait_store_LOW.append(self._sum_intersection_queue) # total number of seconds waited by cars (1 step = 1 second -> 1 car in queue/step = 1 second in queue/step)
            self._avg_intersection_queue_store_LOW.append(self._sum_intersection_queue / self._max_steps) # average number of queued cars per step, in this episode
        if traffic_mode == 2: # data high
            self._reward_store_HIGH.append(tot_neg_reward)
            self._cumulative_wait_store_HIGH.append(self._sum_intersection_queue)
            self._avg_intersection_queue_store_HIGH.append(self._sum_intersection_queue / self._max_steps)
        if traffic_mode == 3: # data ns
            self._reward_store_NS.append(tot_neg_reward)
            self._cumulative_wait_store_NS.append(self._sum_intersection_queue)
            self._avg_intersection_queue_store_NS.append(self._sum_intersection_queue / self._max_steps)
        if traffic_mode == 4: # data ew
            self._reward_store_EW.append(tot_neg_reward)
            self._cumulative_wait_store_EW.append(self._sum_intersection_queue)
            self._avg_intersection_queue_store_EW.append(self._sum_intersection_queue / self._max_steps)

    @property
    def reward_store_LOW(self):
        return self._reward_store_LOW

    @property
    def cumulative_wait_store_LOW(self):
        return self._cumulative_wait_store_LOW

    @property
    def avg_intersection_queue_store_LOW(self):
        return self._avg_intersection_queue_store_LOW

    @property
    def reward_store_HIGH(self):
        return self._reward_store_HIGH

    @property
    def cumulative_wait_store_HIGH(self):
        return self._cumulative_wait_store_HIGH

    @property
    def avg_intersection_queue_store_HIGH(self):
        return self._avg_intersection_queue_store_HIGH

    @property
    def reward_store_NS(self):
        return self._reward_store_NS

    @property
    def cumulative_wait_store_NS(self):
        return self._cumulative_wait_store_NS

    @property
    def avg_intersection_queue_store_NS(self):
        return self._avg_intersection_queue_store_NS

    @property
    def reward_store_EW(self):
        return self._reward_store_EW

    @property
    def cumulative_wait_store_EW(self):
        return self._cumulative_wait_store_EW

    @property
    def avg_intersection_queue_store_EW(self):
        return self._avg_intersection_queue_store_EW

# PLOT AND SAVE THE STATS ABOUT THE SESSION
def save_graphs(object, total_episodes, mode, plot_path):

    plt.rcParams.update({'font.size': 18})
    x = np.linspace(0, total_episodes, math.ceil(total_episodes/4))

    # reward
    if mode == "L":
        data = object.reward_store_LOW
    if mode == "H":
        data = object.reward_store_HIGH
    if mode == "NS":
        data = object.reward_store_NS
    if mode == "EW":
        data = object.reward_store_EW
    plt.plot(x, data)
    plt.ylabel("Cumulative negative reward")
    plt.xlabel("Episode")
    plt.margins(0)
    min_val = min(data)
    max_val = max(data)
    plt.ylim(min_val + 0.05 * min_val, max_val - 0.05 * max_val)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'reward_' + mode + '.png', dpi=96)
    plt.close("all")

    # cumulative wait
    if mode == "L":
        data = object.cumulative_wait_store_LOW
    if mode == "H":
        data = object.cumulative_wait_store_HIGH
    if mode == "NS":
        data = object.cumulative_wait_store_NS
    if mode == "EW":
        data = object.cumulative_wait_store_EW
    plt.plot(x, data)
    plt.ylabel("Cumulative delay (s)")
    plt.xlabel("Episode")
    plt.margins(0)
    min_val = min(data)
    max_val = max(data)
    plt.ylim(min_val - 0.05 * min_val, max_val + 0.05 * max_val)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'delay_' + mode + '.png', dpi=96)
    plt.close("all")

    # average number of cars in queue
    if mode == "L":
        data = object.avg_intersection_queue_store_LOW
    if mode == "H":
        data = object.avg_intersection_queue_store_HIGH
    if mode == "NS":
        data = object.avg_intersection_queue_store_NS
    if mode == "EW":
        data = object.avg_intersection_queue_store_EW
    plt.plot(x, data)
    plt.ylabel("Average queue length (vehicles)")
    plt.xlabel("Episode")
    plt.margins(0)
    min_val = min(data)
    max_val = max(data)
    plt.ylim(min_val - 0.05 * min_val, max_val + 0.05 * max_val)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'queue_' + mode + '.png', dpi=96)
    plt.close("all")


if __name__ == "__main__":

    # --- TRAINING OPTIONS ---
    gui = False
    total_episodes = 250
    gamma = 0.75
    batch_size = 100
    memory_size = 50000
    path = "./model/model_1_5x400_300e_075g/" # nn = 5x400, episodes = 300, gamma = 0.75
    # ----------------------

    # attributes of the agent
    num_states = 80
    num_actions = 4
    max_steps = 5400 # seconds = 1 h 30 min each episode
    green_duration = 10
    yellow_duration = 4

    # sumo mode
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # attributes of the system & inits
    model = Model(num_states, num_actions, batch_size)
    mem = Memory(memory_size)
    sumoCmd = [sumoBinary, "-c", "intersection/tlcs_config_train.sumocfg", "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("PATH:", path)
        print("----- Start time:", datetime.datetime.now())

        sess.run(model.var_init) # start tensorflow
        gr = SimRunner(sess, model, mem, total_episodes, gamma, max_steps, green_duration, yellow_duration, sumoCmd) # init the simulation
        episode = 0

        while episode < total_episodes:
            print('----- Episode {} of {}'.format(episode+1, total_episodes))
            start = timeit.default_timer()

            gr.run(episode) # run the simulation

            stop = timeit.default_timer()
            print('Time: ', round(stop - start, 1))
            episode += 1

        os.makedirs(os.path.dirname(path), exist_ok=True)
        saver.save(sess, path + "my_tlcs_model.ckpt") # save the neural network trained
        print("----- End time:", datetime.datetime.now())
        print("PATH:", path)
        save_graphs(gr, total_episodes, "L", path)
        save_graphs(gr, total_episodes, "H", path)
        save_graphs(gr, total_episodes, "NS", path)
        save_graphs(gr, total_episodes, "EW", path)
