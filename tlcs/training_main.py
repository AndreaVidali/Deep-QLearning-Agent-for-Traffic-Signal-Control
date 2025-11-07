import datetime
import os
import timeit

from rich import print

from tlcs.memory import Memory
from tlcs.model import TrainModel
from tlcs.model_training import replay
from tlcs.training_simulation import Simulation
from tlcs.utils import import_train_configuration, set_sumo
from tlcs.visualization import Visualization


def main() -> None:
    config = import_train_configuration(config_file="TLCS/training_settings.yaml")
    sumo_cmd = set_sumo(config.gui, config.sumocfg_file, config.max_steps)
    os.makedirs(config.models_path, exist_ok=True)

    model = TrainModel(
        config.num_layers,
        config.width_layers,
        config.batch_size,
        config.learning_rate,
        input_dim=config.num_states,
        output_dim=config.num_actions,
    )

    memory = Memory(size_max=config.memory_size_max, size_min=config.memory_size_min)
    visualization = Visualization(config.models_path, dpi=96)

    simulation = Simulation(model=model, memory=memory, sumo_cmd=sumo_cmd, config=config)

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

    model.save_model(config.models_path)

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", config.models_path)

    # Use the instance stores (not class attributes)
    visualization.save_data_and_plot(
        data=simulation.reward_store,
        filename="reward",
        xlabel="Episode",
        ylabel="Cumulative negative reward",
    )
    visualization.save_data_and_plot(
        data=simulation.cumulative_wait_store,
        filename="delay",
        xlabel="Episode",
        ylabel="Cumulative delay (s)",
    )
    visualization.save_data_and_plot(
        data=simulation.avg_queue_length_store,
        filename="queue",
        xlabel="Episode",
        ylabel="Average queue length (vehicles)",
    )


if __name__ == "__main__":
    main()
