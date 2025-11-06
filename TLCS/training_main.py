import datetime
import os
from shutil import copyfile

from generator import TrafficGenerator
from memory import Memory
from model import TrainModel
from training_simulation import Simulation
from utils import import_train_configuration, set_sumo, set_train_path
from visualization import Visualization

if __name__ == "__main__":
    config = import_train_configuration(config_file="TLCS/training_settings.ini")
    sumo_cmd = set_sumo(config["gui"], config["sumocfg_file_name"], config["max_steps"])
    path = set_train_path(config["models_path_name"])

    model = TrainModel(
        config["num_layers"],
        config["width_layers"],
        config["batch_size"],
        config["learning_rate"],
        input_dim=config["num_states"],
        output_dim=config["num_actions"],
    )

    memory = Memory(config["memory_size_max"], config["memory_size_min"])

    traffic_gen = TrafficGenerator(config["max_steps"], config["n_cars_generated"])

    visualization = Visualization(path, dpi=96)

    simulation = Simulation(
        model=model,
        memory=memory,
        traffic_gen=traffic_gen,
        sumo_cmd=sumo_cmd,
        gamma=config["gamma"],
        max_steps=config["max_steps"],
        green_duration=config["green_duration"],
        yellow_duration=config["yellow_duration"],
        num_states=config["num_states"],
        num_actions=config["num_actions"],
        training_epochs=config["training_epochs"],
    )

    episode = 0
    timestamp_start = datetime.datetime.now()

    while episode < config["total_episodes"]:
        print("\n----- Episode", str(episode + 1), "of", str(config["total_episodes"]))
        # set the epsilon for this episode according to epsilon-greedy policy
        epsilon = 1.0 - (episode / config["total_episodes"])
        # run the simulation
        simulation_time, training_time = simulation.run(episode, epsilon)
        print(
            "Simulation time:",
            simulation_time,
            "s - Training time:",
            training_time,
            "s - Total:",
            round(simulation_time + training_time, 1),
            "s",
        )
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    model.save_model(path)

    copyfile(
        src="training_settings.ini", dst=os.path.join(path, "training_settings.ini")
    )

    visualization.save_data_and_plot(
        data=Simulation.reward_store,
        filename="reward",
        xlabel="Episode",
        ylabel="Cumulative negative reward",
    )
    visualization.save_data_and_plot(
        data=Simulation.cumulative_wait_store,
        filename="delay",
        xlabel="Episode",
        ylabel="Cumulative delay (s)",
    )
    visualization.save_data_and_plot(
        data=Simulation.avg_queue_length_store,
        filename="queue",
        xlabel="Episode",
        ylabel="Average queue length (vehicles)",
    )
