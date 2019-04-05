# created by Andrea Vidali
# info@andreavidali.com

from __future__ import absolute_import
from __future__ import print_function

from routes_evaluate import generate_routes_evaluate

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

# sumo things - we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


class SimRunner:
    def __init__(self, max_steps, green_sec_s_dx, green_sec_sx, yellow_sec):
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
        self._max_steps = max_steps
        self._green_duration_s_dx = green_sec_s_dx
        self._green_duration_sx = green_sec_sx
        self._yellow_duration = yellow_sec


    def run(self, sumoCmd, routefile_seed, traffic_mode, traffic_distribution):
        # generate the route file for this simulation
        self._n_cars_generated, self._car_gen_step = generate_routes_evaluate(routefile_seed, traffic_mode, traffic_distribution, self._max_steps)
        traci.start(sumoCmd)

        self._steps = 0
        old_wait_time = 0

        while self._steps < self._max_steps:
            if self._light_phase == 0 or self._light_phase == 4:
                current_wait_time = self._simulate_phase(self._green_duration_s_dx)
            if self._light_phase == 2 or self._light_phase == 6:
                current_wait_time = self._simulate_phase(self._green_duration_sx)
            if self._light_phase % 2 != 0:
                current_wait_time = self._simulate_phase(self._yellow_duration)

            #  save reward to compare with the agent
            reward = old_wait_time - current_wait_time
            self._reward_store.append(reward)

            # activate the next phase
            self._select_next_light_phase()

            # saving the variables for the next step
            old_wait_time = current_wait_time
            if reward < 0:
                self._tot_reward += reward

        print("Total negative reward: {}".format(self._tot_reward))
        traci.close()

    def _simulate_phase(self, steps_todo):
        traci.trafficlight.setPhase("TL", self._light_phase)
        if (self._steps + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._steps
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


def save_graphs(object, plot_path, index_plot, plot_reward_store, plot_summed_wait, plot_troughput, plot_queue):
    plt.rcParams.update({'font.size': 18})

    data = plot_reward_store[index_plot] # reward
    plt.plot(data)
    plt.ylabel("Istant reward")
    plt.xlabel("n-th action")
    plt.margins(0)
    min_val = min(data)
    max_val = max(data)
    plt.ylim(min_val + 0.05 * min_val, max_val + 0.05 * max_val)
    plt.xlim(0, len(data))
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'reward.png', dpi=96)
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
    traffic_mode = "low" # low, high, ns, ew
    traffic_distribution = "weibull" # weibull, geometric, uniform, triangular, gamma, beta
    gui = False
    path = "./model/model_static/"
    max_steps = 5400
    green_sec_s_dx = 30
    green_sec_sx = 15
    yellow_sec = 4
    # -----------------------

    # sumo mode
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # attributes of the system & init
    routefile_seed = 3500 # seed for random generation
    num_states = 80
    num_actions = 4
    sumoCmd = [sumoBinary, "-c", "intersection/tlcs_config_evaluate.sumocfg", "--no-step-log", "true"]
    plot_path = path + "AVG_eval_" + traffic_distribution + "_" + traffic_mode + "/"
    iterations = 5
    current_iteration = 0

    # lists for saving the stats and then avg them
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
        gr = SimRunner(max_steps, green_sec_s_dx, green_sec_sx, yellow_sec)
        gr.run(sumoCmd, routefile_seed, traffic_mode, traffic_distribution)

        # save data about this epsiode
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

    # print the data aggregated
    print("-----------------")
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

    # save the data printed before
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
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

    # save some graphs
    index_plot = np.argsort(eval_rew)[len(eval_rew)//2] # which episode the data comes from -> the epsiode with the median reward
    save_graphs(gr, plot_path, index_plot, plot_reward_store, plot_summed_wait, plot_troughput, plot_queue)
