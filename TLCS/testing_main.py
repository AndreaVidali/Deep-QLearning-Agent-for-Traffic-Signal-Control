import os
from shutil import copyfile

from model import TestModel
from testing_simulation import Simulation
from utils import import_test_configuration, set_sumo, set_test_path
from visualization import Visualization

if __name__ == "__main__":
    config = import_test_configuration(config_file="TLCS/testing_settings.yaml")
    sumo_cmd = set_sumo(config.gui, config.sumocfg_file, config.max_steps)
    model_path, plot_path = set_test_path(config.models_path, config.model_to_test)

    model = TestModel(input_dim=config.num_states, model_path=model_path)

    visualization = Visualization(plot_path, dpi=96)

    simulation = Simulation(model=model, sumo_cmd=sumo_cmd, config=config)

    print("\n----- Test episode")
    simulation_time = simulation.run(config["episode_seed"])  # run the simulation
    print("Simulation time:", simulation_time, "s")

    print("----- Testing info saved at:", plot_path)

    copyfile(src="testing_settings.ini", dst=os.path.join(plot_path, "testing_settings.ini"))

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
