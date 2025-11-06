import os
from shutil import copyfile

from model import TestModel
from testing_simulation import Simulation
from utils import import_test_configuration, set_sumo, set_test_path
from visualization import Visualization

if __name__ == "__main__":
    config = import_test_configuration(config_file="TLCS/testing_settings.ini")
    sumo_cmd = set_sumo(config["gui"], config["sumocfg_file_name"], config["max_steps"])
    model_path, plot_path = set_test_path(
        config["models_path_name"], config["model_to_test"]
    )

    model = TestModel(input_dim=config["num_states"], model_path=model_path)

    visualization = Visualization(plot_path, dpi=96)

    simulation = Simulation(
        model=model,
        sumo_cmd=sumo_cmd,
        max_steps=config["max_steps"],
        n_cars_generated=config["n_cars_generated"],
        green_duration=config["green_duration"],
        yellow_duration=config["yellow_duration"],
        num_states=config["num_states"],
    )

    print("\n----- Test episode")
    simulation_time = simulation.run(config["episode_seed"])  # run the simulation
    print("Simulation time:", simulation_time, "s")

    print("----- Testing info saved at:", plot_path)

    copyfile(
        src="testing_settings.ini", dst=os.path.join(plot_path, "testing_settings.ini")
    )

    visualization.save_data_and_plot(
        data=Simulation.reward_episode,
        filename="reward",
        xlabel="Action step",
        ylabel="Reward",
    )
    visualization.save_data_and_plot(
        data=Simulation.queue_length_episode,
        filename="queue",
        xlabel="Step",
        ylabel="Queue lenght (vehicles)",
    )
