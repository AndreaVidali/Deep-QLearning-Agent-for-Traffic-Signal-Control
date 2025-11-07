from pathlib import Path
from shutil import copyfile

from tlcs.model import TestModel
from tlcs.plots import save_data_and_plot
from tlcs.testing_simulation import Simulation
from tlcs.utils import import_test_configuration, set_sumo, set_test_path

CONFIG_FILE = Path("tlcs/testing_settings.yaml")

if __name__ == "__main__":
    config = import_test_configuration(CONFIG_FILE)
    sumo_cmd = set_sumo(config.gui, config.sumocfg_file, config.max_steps)
    model_path, plot_path = set_test_path(config.models_path, config.model_to_test)

    model = TestModel(input_dim=config.num_states, model_path=model_path)

    simulation = Simulation(model=model, sumo_cmd=sumo_cmd, config=config)

    print("\n----- Test episode")
    simulation_time = simulation.run(config.episode_seed)  # run the simulation
    print("Simulation time:", simulation_time, "s")

    print("----- Testing info saved at:", plot_path)

    copyfile(src="testing_settings.yaml", dst=plot_path / "testing_settings.yaml")

    save_data_and_plot(
        data=Simulation.reward_episode,
        filename="reward",
        xlabel="Action step",
        ylabel="Reward",
        out_folder=config.models_path,
    )
    save_data_and_plot(
        data=Simulation.queue_length_episode,
        filename="queue",
        xlabel="Step",
        ylabel="Queue lenght (vehicles)",
        out_folder=config.models_path,
    )
