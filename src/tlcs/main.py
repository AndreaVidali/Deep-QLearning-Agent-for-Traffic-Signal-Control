import datetime
import timeit
from pathlib import Path
from shutil import copyfile

from rich import print

from tlcs.memory import Memory
from tlcs.model import TestModel, TrainModel
from tlcs.model_training import replay
from tlcs.plots import save_data_and_plot
from tlcs.testing_simulation import TestingSimulation
from tlcs.training_simulation import TrainingSimulation
from tlcs.utils import import_test_configuration, import_train_configuration, set_sumo


def training_session(config_file: Path, out_path: Path) -> None:
    config = import_train_configuration(config_file)
    sumo_cmd = set_sumo(
        gui=config.gui,
        sumocfg_file=config.sumocfg_file,
        max_steps=config.max_steps,
    )
    out_path.mkdir(parents=True, exist_ok=True)

    model = TrainModel(
        config.num_layers,
        config.width_layers,
        config.batch_size,
        config.learning_rate,
        input_dim=config.num_states,
        output_dim=config.num_actions,
    )

    memory = Memory(size_max=config.memory_size_max, size_min=config.memory_size_min)

    simulation = TrainingSimulation(model=model, memory=memory, sumo_cmd=sumo_cmd, config=config)

    episode = 0
    timestamp_start = datetime.datetime.now()
    tot_episodes = config.total_episodes

    while episode < tot_episodes:
        print(f"\n----- Episode {episode + 1} of {tot_episodes}")

        # set the epsilon for this episode according to epsilon-greedy policy
        epsilon = 1.0 - (episode / tot_episodes)

        # run the simulation
        simulation_time = simulation.run(episode, epsilon)

        # train the model
        start_time = timeit.default_timer()
        for _ in range(config.training_epochs):
            replay(
                model=model,
                memory=memory,
                gamma=config.gamma,
                num_states=config.num_states,
                num_actions=config.num_actions,
            )
        training_time = round(timeit.default_timer() - start_time, 1)

        episode += 1

        print(
            f"Simulation time: {simulation_time} s | "
            f"Training time: {training_time} s | "
            f"Total: {round(simulation_time + training_time, 1)} s",
        )

    model.save_model(out_path)

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", out_path)

    save_data_and_plot(
        data=simulation.reward_store,
        filename="reward",
        xlabel="Episode",
        ylabel="Cumulative negative reward",
        out_folder=out_path,
    )
    save_data_and_plot(
        data=simulation.cumulative_wait_store,
        filename="delay",
        xlabel="Episode",
        ylabel="Cumulative delay (s)",
        out_folder=out_path,
    )
    save_data_and_plot(
        data=simulation.avg_queue_length_store,
        filename="queue",
        xlabel="Episode",
        ylabel="Average queue length (vehicles)",
        out_folder=out_path,
    )


def testing_session(config_file: Path, model_path: Path) -> None:
    config = import_test_configuration(config_file)
    sumo_cmd = set_sumo(config.gui, config.sumocfg_file, config.max_steps)
    tests_path = model_path / "tests"

    model = TestModel(input_dim=config.num_states, model_path=model_path)

    simulation = TestingSimulation(model=model, sumo_cmd=sumo_cmd, config=config)

    print("\n----- Test episode")
    simulation_time = simulation.run(config.episode_seed)
    print("Simulation time:", simulation_time, "s")

    print("----- Testing info saved at:", tests_path)

    copyfile(src=config_file, dst=tests_path / config_file)

    save_data_and_plot(
        data=simulation.reward_episode,
        filename="reward",
        xlabel="Action step",
        ylabel="Reward",
        out_folder=tests_path,
    )
    save_data_and_plot(
        data=simulation.queue_length_episode,
        filename="queue",
        xlabel="Step",
        ylabel="Queue lenght (vehicles)",
        out_folder=tests_path,
    )
