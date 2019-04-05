# created by Andrea Vidali
# info@andreavidali.com

from __future__ import absolute_import
from __future__ import print_function

from routes_train import generate_routes_train

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

# phase codes based on xai_tlcs.net.xml
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

# CLASS OF THE NEURAL NETWORK
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
        fc1 = tf.layers.dense(self._states, 400, activation=tf.nn.leaky_relu)
        fc2 = tf.layers.dense(fc1, 400, activation=tf.nn.leaky_relu)
        fc3 = tf.layers.dense(fc2, 400, activation=tf.nn.leaky_relu)
        fc4 = tf.layers.dense(fc3, 400, activation=tf.nn.leaky_relu)
        fc5 = tf.layers.dense(fc4, 400, activation=tf.nn.leaky_relu)
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
    def __init__(self, sess, model, memory, green_sec, yellow_sec, gamma, max_steps, sumoCmd):
        self._sess = sess
        self._model = model
        self._memory = memory
        self._eps = 0 # controls the explorative/exploitative payoff
        self._green_duration = green_sec
        self._yellow_duration = yellow_sec
        self._gamma = gamma
        self._max_steps = max_steps
        self._sumoCmd = sumoCmd

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
    def run(self, episode, total_episodes):
        # first, generate the route file for this simulation and set up sumo
        traffic_code = generate_routes_train(episode, self._max_steps)
        traci.start(self._sumoCmd)

        # set the epsilon for this episode
        self._eps = 1.0 - (episode / total_episodes)

        # inits
        self._steps = 0
        self._sum_intersection_queue = 0
        tot_neg_reward = 0
        old_wait_time = 0
        save = False

        # simulation (self._steps updated in function "simulate")
        while self._steps < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # choose the action to perform based on the current state
            action = self._choose_action(current_state)

            # if the chosen action is different from the last one, activate the yellow phase
            if self._steps != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                current_wait_time = self._simulate(self._yellow_duration)

            # execute the action selected before
            self._set_green_phase(action)
            current_wait_time = self._simulate(self._green_duration)

            #  calculate reward
            reward = old_wait_time - current_wait_time

            # data saving into memory
            if save == True:
                self._memory.add_sample((old_state, old_action, reward, current_state))

            # saving the variables for the next step & accumulate reward
            old_state = current_state
            old_action = action
            old_wait_time = current_wait_time
            save = True
            if reward < 0:
                tot_neg_reward += reward

        self._save_stats(traffic_code, tot_neg_reward)

        print("Total negative reward: {}, Eps: {}".format(tot_neg_reward, self._eps))
        traci.close()

    # HANDLE THE CORRECT NUMBER OF STEPS TO SIMULATE
    def _simulate(self, steps_todo):
        intersection_queue, summed_wait = self._get_stats() # init the summed_wait, in order to avoid a null return
        if (self._steps + steps_todo) >= self._max_steps: # do not do more steps than the maximum number of steps
            steps_todo = self._max_steps - self._steps
        while steps_todo > 0:
            traci.simulationStep() # simulate 1 step in sumo
            self._steps = self._steps + 1
            steps_todo -= 1
            self._replay() # training
            intersection_queue, summed_wait = self._get_stats()
            self._sum_intersection_queue += intersection_queue
        return summed_wait

    # RETRIEVE THE STATS OF THE SIMULATION FOR ONE SINGLE STEP
    def _get_stats(self):
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL") # number of cars in halt in a road
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        intersection_queue = halt_N + halt_S + halt_E + halt_W

        wait_N = traci.edge.getWaitingTime("N2TL") # total waiting times of cars in a road
        wait_S = traci.edge.getWaitingTime("S2TL")
        wait_W = traci.edge.getWaitingTime("E2TL")
        wait_E = traci.edge.getWaitingTime("W2TL")
        summed_wait = wait_N + wait_S + wait_W + wait_E

        return intersection_queue, summed_wait

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

    # RETRIEVE THE STATE OF THE INTERSECTION FROM SUMO
    def _get_state(self):
        state = np.zeros(self._model.num_states)

        for veh_id in traci.vehicle.getIDList():
            lane_pos = traci.vehicle.getLanePosition(veh_id)
            lane_id = traci.vehicle.getLaneID(veh_id)
            lane_pos = 750 - lane_pos # inversion of lane so if the car is close to TL, lane_pos = 0
            lane_group = -1 # dummy initialization
            valid_car = False # flag for not detecting cars that are crossing the intersection or driving away from it

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

            # in which lane is the car? "x2TL_3" are the "turn left only" lanes
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
        if len(batch) > 0: # if there is at least 1 sample in the memory
            states = np.array([val[0] for val in batch]) # isolate the old states from the batch
            next_states = np.array([val[3] for val in batch]) # isolate the next states from the batch
            q_s_a = self._model.predict_batch(states, self._sess) # predict Q-values starting from the old states
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
    def _save_stats(self, traffic_code, tot_neg_reward): # save the stats for this episode
        if traffic_code == 1: # data low
            self._reward_store_LOW.append(tot_neg_reward) # how much negative reward in this episode
            self._cumulative_wait_store_LOW.append(self._sum_intersection_queue) # total number of seconds waited by cars (1 step = 1 second -> 1 car in queue/step = 1 second in queue/step=
            self._avg_intersection_queue_store_LOW.append(self._sum_intersection_queue / self._max_steps) # average number of queued cars per step, in this episode

        if traffic_code == 2: # data high
            self._reward_store_HIGH.append(tot_neg_reward)
            self._cumulative_wait_store_HIGH.append(self._sum_intersection_queue)
            self._avg_intersection_queue_store_HIGH.append(self._sum_intersection_queue / self._max_steps)

        if traffic_code == 3: # data ns
            self._reward_store_NS.append(tot_neg_reward)
            self._cumulative_wait_store_NS.append(self._sum_intersection_queue)
            self._avg_intersection_queue_store_NS.append(self._sum_intersection_queue / self._max_steps)

        if traffic_code == 4: # data ew
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
    plt.xlabel("Epoch")
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
    plt.xlabel("Epoch")
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
    plt.xlabel("Epoch")
    plt.margins(0)
    min_val = min(data)
    max_val = max(data)
    plt.ylim(min_val - 0.05 * min_val, max_val + 0.05 * max_val)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'queue_' + mode + '.png', dpi=96)
    plt.close("all")


if __name__ == "__main__":

    # --- OPTIONS ---
    gui = False
    total_episodes = 16
    batch_size = 100
    memory_size = 50000
    gamma = 0.75 # future discount
    path = "./model/model_2_5x400_300e_075g/" # nn = 5x400, episodes = 300, gamma = 0.75
    # ----------------------

    # attributes of the agent
    num_states = 80
    num_actions = 4
    green_sec = 10 # duration of green phase
    yellow_sec = 4 # duration of yellow phase
    max_steps = 5400 # seconds - simulation is 1 h 30 min long

    # sumo mode
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # attributes of the system & inits
    model = Model(num_states, num_actions, batch_size)
    mem = Memory(memory_size)
    sumoCmd = [sumoBinary, "-c", "intersection/tlcs_config_train.sumocfg", "--no-step-log", "true"]
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("PATH:", path)
        print("----- Start time:", datetime.datetime.now())

        sess.run(model.var_init) # start tensorflow
        gr = SimRunner(sess, model, mem, green_sec, yellow_sec, gamma, max_steps, sumoCmd) # init the simulation
        episode = 0

        while episode < total_episodes:
            print('----- Epoch {} of {}'.format(episode+1, total_episodes))
            start = timeit.default_timer()

            gr.run(episode, total_episodes) # run the simulation

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
