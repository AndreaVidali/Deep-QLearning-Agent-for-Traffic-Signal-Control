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
from tlcs.utils import load_testing_settings, load_training_settings, set_sumo


def training_session(settings_file: Path, out_path: Path) -> None:
    settings = load_training_settings(settings_file)
    sumo_cmd = set_sumo(
        gui=settings.gui,
        sumocfg_file=settings.sumocfg_file,
        max_steps=settings.max_steps,
    )
    out_path.mkdir(parents=True, exist_ok=True)

    model = TrainModel(
        settings.num_layers,
        settings.width_layers,
        settings.batch_size,
        settings.learning_rate,
        input_dim=settings.num_states,
        output_dim=settings.num_actions,
    )

    memory = Memory(size_max=settings.memory_size_max, size_min=settings.memory_size_min)

    simulation = TrainingSimulation(
        model=model,
        memory=memory,
        sumo_cmd=sumo_cmd,
        settings=settings,
    )

    episode = 0
    timestamp_start = datetime.datetime.now()
    tot_episodes = settings.total_episodes

    while episode < tot_episodes:
        print(f"\n----- Episode {episode + 1} of {tot_episodes}")

        # set the epsilon for this episode according to epsilon-greedy policy
        epsilon = 1.0 - (episode / tot_episodes)

        # run the simulation
        simulation_time = simulation.run(episode, epsilon)

        # train the model
        start_time = timeit.default_timer()
        for _ in range(settings.training_epochs):
            replay(
                model=model,
                memory=memory,
                gamma=settings.gamma,
                num_states=settings.num_states,
                num_actions=settings.num_actions,
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

    copyfile(src=settings_file, dst=out_path / settings_file)

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


def testing_session(settings_file: Path, model_path: Path) -> None:
    settings = load_testing_settings(settings_file)
    sumo_cmd = set_sumo(settings.gui, settings.sumocfg_file, settings.max_steps)
    tests_path = model_path / "tests"

    model = TestModel(input_dim=settings.num_states, model_path=model_path)

    simulation = TestingSimulation(model=model, sumo_cmd=sumo_cmd, settings=settings)

    print("\n----- Test episode")
    simulation_time = simulation.run(settings.episode_seed)
    print("Simulation time:", simulation_time, "s")

    print("----- Testing info saved at:", tests_path)

    copyfile(src=settings_file, dst=tests_path / settings_file)

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
