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
import matplotlib.pyplot as plt
from collections import Counter
import scipy.stats as stats
import bottleneck as bn


os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # to kill warning about tf
import tensorflow as tf


MAX_STEPS = 5400 # seconds - simulation is 1 h 30 min long

GREEN_PHASE_SEC = 10
YELLOW_PHASE_SEC = 4

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

# generation of the routes of cars
def generate_routefile(seed, mode, distribution):
    np.random.seed(seed)  # make tests reproducible

    low_mode = False
    standard_mode = False
    NS_mode = False
    EW_mode = False

    if mode == "low": # low density
        n_cars_generated = 600
        low_mode = True
        print("Mode: low")
    elif mode == "high": # standard experiment
        n_cars_generated = 4000
        standard_mode = True
        print("Mode: high")
    elif mode == "ns": # main source is north/south
        n_cars_generated = 2000
        NS_mode = True
        print("Mode: north-south main")
    elif mode == "ew":  # main source is east/west
        n_cars_generated = 2000
        EW_mode = True
        print("Mode: east-west main")

    if distribution == "weibull":
        timings = np.random.weibull(2, n_cars_generated) # high traffic in beginning-middle
    if distribution == "geometric":
        timings = np.random.geometric(0.1, n_cars_generated) # super high at beginning
    if distribution == "uniform":
        timings = np.random.uniform(-1, 0, n_cars_generated)
    if distribution == "triangular":
        timings = np.random.triangular(-3, 6, 8, n_cars_generated) # triangle peak towards the end
    if distribution == "gamma":
        timings = np.random.standard_gamma(3, n_cars_generated) # peak at beginning
    if distribution == "beta":
        timings = np.random.beta(0.5, 0.5, n_cars_generated) # two peaks at beginning and end

    # rescale the distribution to fit the step interval 0:MAX_STEPS
    timings = np.sort(timings)
    car_gen_steps = []
    min_old = math.floor(timings[1])
    max_old = math.ceil(timings[-1])
    min_new = 0
    max_new = MAX_STEPS
    for value in timings:
        car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

    car_gen_steps = np.rint(car_gen_steps) # effective steps when a car will be generated

    with open("intersection/tlcs_evaluate.rou.xml", "w") as routes:
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
                if straight_or_turn < 0.75: # cars that go straight
                    route_straight = np.random.randint(1, 5) # choose a random source road
                    if route_straight == 1:
                        print('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 2:
                        print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 3:
                        print('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    else:
                        print('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                else: # car that turn
                    route_turn = np.random.randint(1, 9) # choose random source road and destination road
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
                straight_or_turn = np.random.uniform()
                source = np.random.uniform()
                destination_straight = np.random.uniform()
                destination_turn = np.random.randint(1, 5)
                if straight_or_turn < 0.75: # choose behavior: straight or turn
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

        return n_cars_generated, car_gen_steps


class SimRunner:
    def __init__(self):
        self._steps = 0
        self._reward_store = []
        self._intersection_queue_store = []
        self._summed_wait_store = []
        self._throughput_store = []
        self._cumulative_arrived_now = 0
        self._carID_that_waited = []
        self._n_cars_generated = 0
        self._car_gen_step = []
        self._carID_sx = []
        self._carID_aheaddx = []
        self._tot_reward = 0
        self._light_phase = 0


    def run(self, sumoCmd, routefile_seed, traffic_mode, traffic_distribution):
        # first, generate the route file for this simulation
        self._n_cars_generated, self._car_gen_step = generate_routefile(routefile_seed, traffic_mode, traffic_distribution)
        traci.start(sumoCmd)

        self._steps = 0
        tot_reward = 0
        old_wait_time = 0
        summed_wait = 0

        green_sdx = 30
        green_sx = 10
        yellow = 4

        while self._steps < MAX_STEPS:
            #  calculate reward: (change in cumulative reward between actions)
            current_wait_time = summed_wait # this is the situation after the action (last scan of the intersection)
            reward = old_wait_time - current_wait_time

            # get current state of the intersection
            if self._light_phase == 0 or self._light_phase == 4:
                summed_wait = self._simulate_phase(green_sdx)
            if self._light_phase == 2 or self._light_phase == 6:
                summed_wait = self._simulate_phase(green_sx)
            if self._light_phase % 2 != 0:
                summed_wait = self._simulate_phase(yellow)

            self._select_next_light_phase()

            self._reward_store.append(reward)

            # saving the variables for the next step
            old_wait_time = current_wait_time
            if reward < 0:
                self._tot_reward += reward


        print("Total negative reward: {}".format(self._tot_reward))
        traci.close()

    def _simulate_phase(self, PHASE_SEC):
        traci.trafficlight.setPhase("TL", self._light_phase)
        steps_todo = self._calculate_steps(PHASE_SEC)
        self._steps = self._steps + steps_todo
        while steps_todo > 0:
            traci.simulationStep()
            steps_todo -= 1
            # stats
            intersection_queue, summed_wait, arrived_now = self._get_stats()
            self._intersection_queue_store.append(intersection_queue)
            self._summed_wait_store.append(summed_wait)
            self._cumulative_arrived_now += arrived_now
            self._throughput_store.append(self._cumulative_arrived_now)
        return summed_wait

    def _select_next_light_phase(self):
        self._light_phase += 1
        if self._light_phase == 8:
            self._light_phase = 0

    def _calculate_steps(self, phase_steps):
        # check if the steps to do is over the MAX_STEPS
        if (self._steps + phase_steps) >= MAX_STEPS:
            phase_steps = MAX_STEPS - self._steps
        return phase_steps

    def _get_stats(self):
        route_turnleft = {"W_N", "N_E", "E_S", "S_W"}
        intersection_queue = 0
        summed_wait = 0
        for veh_id in traci.vehicle.getIDList():
            wait_time_car = traci.vehicle.getWaitingTime(veh_id)
            if wait_time_car > 0.5:
                intersection_queue += 1
                self._carID_that_waited.append(veh_id)
                route_ID = traci.vehicle.getRouteID(veh_id)
                if route_ID in route_turnleft:
                    self._carID_sx.append(veh_id)
                else:
                    self._carID_aheaddx.append(veh_id)
            summed_wait += wait_time_car
        arrived_now = traci.simulation.getArrivedNumber()
        return intersection_queue, summed_wait, arrived_now

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def summed_wait_store(self):
        return self._summed_wait_store

    @property
    def throughput_store(self):
        return self._throughput_store

    @property
    def intersection_queue_store(self):
        return self._intersection_queue_store

    @property
    def carID_that_waited(self):
        return self._carID_that_waited

    @property
    def n_cars_generated(self):
        return self._n_cars_generated

    @property
    def car_gen_steps(self):
        return self._car_gen_step

    @property
    def carID_sx(self):
        return self._carID_sx

    @property
    def carID_aheaddx(self):
        return self._carID_aheaddx

    @property
    def tot_reward(self):
        return self._tot_reward


def plot_stats(object, plot_path, index_plot, plot_reward_store, plot_summed_wait, plot_troughput, plot_queue):
    plt.rcParams.update({'font.size': 18})

    data = plot_reward_store[index_plot]
    plt.plot(data) # reward
    plt.ylabel("Istant reward")
    plt.xlabel("n-th action")
    plt.margins(0)
    min_val = min(data)
    max_val = max(data)
    plt.ylim(min_val + 0.05 * min_val, max_val + 0.05 * max_val)
    plt.xlim(0, len(data))
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'reward.png', dpi=96)#plt.show()
    plt.close("all")

    data = plot_summed_wait[index_plot] # cumulative delay
    data = bn.move_mean(data, window=25, min_count=1)
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_ylabel("Cumulative delay", color=color)
    ax1.set_xlabel("Step")
    ax1.plot(data, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.margins(0)
    min_val = min(data)
    max_val = max(data)
    ax1.set_ylim([0, max_val + 0.05 * max_val])
    ax1.set_xlim([0, 5400])
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Number of cars generated', color=color)
    ax2.hist(gr.car_gen_steps, bins=25, histtype='stepfilled', alpha=0.18, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_xlim([0, 5400])
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'delay.png', dpi=96)
    plt.close("all")

    data = plot_troughput[index_plot] # num of cars arrived at the end
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_ylabel("Throughput (vehicles)", color=color)
    ax1.set_xlabel("Step")
    ax1.plot(data, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.margins(0)
    min_val = min(data)
    max_val = max(data)
    ax1.set_ylim([0, max_val + 0.05 * max_val])
    ax1.set_xlim([0, 5400])
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Number of cars generated', color=color)
    ax2.hist(gr.car_gen_steps, bins=25, histtype='stepfilled', alpha=0.18, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_xlim([0, 5400])
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'tp.png', dpi=96)
    plt.close("all")

    data = plot_queue[index_plot] # average number of cars in queue
    data = bn.move_mean(data, window=50, min_count=1)
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_ylabel("Number of cars in queue", color=color)
    ax1.set_xlabel("Step")
    ax1.plot(data, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.margins(0)
    min_val = min(data)
    max_val = max(data)
    ax1.set_ylim([0, max_val + 0.05 * max_val])
    ax1.set_xlim([0, 5400])
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Number of cars generated', color=color)
    ax2.hist(gr.car_gen_steps, bins=25, histtype='stepfilled', alpha=0.18, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_xlim([0, 5400])
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'queue.png', dpi=96)
    plt.close("all")


def find_avg_waiting_times(dict):
    sum = 0
    i = 0
    car_wait_list = Counter(dict)
    for key, value in car_wait_list.items():
        sum = sum + value
        i = i + 1
    avg_wait_car = sum / i
    return avg_wait_car


def find_std_waiting_times(dict):
    list = []
    car_wait_list = Counter(dict)
    for key, value in car_wait_list.items():
        list.append(value)
    return np.std(list)


if __name__ == "__main__":

    # --- TESTING OPTIONS ---
    routefile_seed = 3500 # seed for random generation
    traffic_mode = "low" # low, high, ns, ew
    traffic_distribution = "weibull" # weibull, geometric, uniform, triangular, gamma, beta
    gui = False
    plot = True
    path = "./model/model_static/"
    plot_path = path + "AVG_eval_" + traffic_distribution + "_" + traffic_mode + "/"
    # -----------------------

    # sumo mode
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # attributes of the system & init
    num_states = 80
    num_actions = 4
    sumoCmd = [sumoBinary, "-c", "intersection/tlcs_config_evaluate.sumocfg", "--no-step-log", "true"]

    # lists for saving the stats and then avg them
    iterations = 5
    current_iteration = 0
    eval_rew = []
    eval_twt = []
    eval_avgQ = []
    eval_carsGen = []
    eval_carsWait = []
    eval_carsWait_SDX = []
    eval_carsWait_SX = []
    eval_avgW = []
    eval_avgW_std = []
    eval_avgW_SDX = []
    eval_avgW_SDX_std = []
    eval_avgW_SX = []
    eval_avgW_SX_std = []
    plot_reward_store = []
    plot_summed_wait = []
    plot_troughput = []
    plot_queue = []


    print("----- Static Traffic Light")
    print("PATH:", path)
    while current_iteration < iterations:
        gr = SimRunner()
        gr.run(sumoCmd, routefile_seed, traffic_mode, traffic_distribution)

        eval_rew.append(gr.tot_reward)
        eval_twt.append(sum(gr.intersection_queue_store))
        eval_avgQ.append(round(np.mean(gr.intersection_queue_store), 3))
        eval_carsGen.append(gr.n_cars_generated)
        eval_carsWait.append(len(Counter(gr.carID_that_waited).keys()))
        eval_carsWait_SDX.append(len(Counter(gr.carID_aheaddx).keys()))
        eval_carsWait_SX.append(len(Counter(gr.carID_sx).keys()))
        eval_avgW.append(round(find_avg_waiting_times(gr.carID_that_waited), 3))
        eval_avgW_std.append(round(find_std_waiting_times(gr.carID_that_waited), 3))
        eval_avgW_SDX.append(round(find_avg_waiting_times(gr.carID_aheaddx), 3))
        eval_avgW_SDX_std.append(round(find_std_waiting_times(gr.carID_aheaddx), 3))
        eval_avgW_SX.append(round(find_avg_waiting_times(gr.carID_sx), 3))
        eval_avgW_SX_std.append(round(find_std_waiting_times(gr.carID_sx), 3))

        plot_reward_store.append(gr.reward_store)
        plot_summed_wait.append(gr.summed_wait_store)
        plot_troughput.append(gr.throughput_store)
        plot_queue.append(gr.intersection_queue_store)

        current_iteration = current_iteration + 1
        routefile_seed = routefile_seed + 1

    print("-----------------")
    #print("Average summed delay /step:", round(np.mean(gr.summed_wait_store), 3))
    print("- Average negative reward:", round(np.mean(eval_rew), 3))
    print("- Total wait time:", round(np.mean(eval_twt), 3))
    print("- Average queue length /step:", round(np.mean(eval_avgQ), 3))
    print("- Total cars generated:", round(np.mean(eval_carsGen), 3))
    print("--- Waiting cars:", round(np.mean(eval_carsWait), 3))
    print("--- Waiting cars [straight or dx]:", round(np.mean(eval_carsWait_SDX), 3))
    print("--- Waiting cars [sx]:", round(np.mean(eval_carsWait_SX), 3))
    print("- Average wait time /car:", round(np.mean(eval_avgW), 3))
    print("--- Average wait time /car [straight or dx]:", round(np.mean(eval_avgW_SDX), 3))
    print("--- Average wait time /car [sx]:", round(np.mean(eval_avgW_SX), 3))

    with open(plot_path + "data.txt", "w") as data:
        print("%s" % (int(round(np.mean(eval_rew), 2))), file=data)
        print("%s" % (int(round(np.mean(eval_twt), 2))), file=data)
        print("%s" % (round(np.mean(eval_avgQ), 1)), file=data)
        print("%s" % (int(round(np.mean(eval_carsGen), 2))), file=data)
        print("%s" % (int(round(np.mean(eval_carsWait), 2))), file=data)
        print("%s" % (int(round(np.mean(eval_carsWait_SDX), 2))), file=data)
        print("%s" % (int(round(np.mean(eval_carsWait_SX), 2))), file=data)
        print("%s" % (round(np.mean(eval_avgW), 1)), file=data)
        print("%s" % (round(np.mean(eval_avgW_SDX), 1)), file=data)
        print("%s" % (round(np.mean(eval_avgW_SX), 1)), file=data)

    if (plot == True):
        index_plot = np.argsort(eval_rew)[len(eval_rew)//2]
        plot_stats(gr, plot_path, index_plot, plot_reward_store, plot_summed_wait, plot_troughput, plot_queue)

    with open(plot_path + "plot_data_reward.txt", "w") as f:
        plot_data = plot_reward_store[index_plot]
        for element in plot_data:
            f.write("%s\n" % element)
            #print("%s" % element, file=data)

    with open(plot_path + "plot_data_delay.txt", "w") as f:
        plot_data = plot_summed_wait[index_plot]
        for element in plot_data:
            f.write("%s\n" % element)
            #print("%s" % element, file=data)

    with open(plot_path + "plot_data_tp.txt", "w") as f:
        plot_data = plot_troughput[index_plot]
        for element in plot_data:
            f.write("%s\n" % element)
            #print("%s" % element, file=data)

    with open(plot_path + "plot_data_queue.txt", "w") as f:
        plot_data = plot_queue[index_plot]
        for element in plot_data:
            f.write("%s\n" % element)
            #print("%s" % element, file=data)
