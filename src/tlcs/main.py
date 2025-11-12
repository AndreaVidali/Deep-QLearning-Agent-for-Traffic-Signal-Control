from datetime import datetime
from pathlib import Path
from shutil import copyfile

from tlcs.agent import Agent
from tlcs.constants import TESTING_SETTINGS_FILE, TRAINING_SETTINGS_FILE
from tlcs.env import Environment, EnvStats
from tlcs.episode import Record, run_episode
from tlcs.memory import Memory
from tlcs.plots import save_data_and_plot
from tlcs.settings import load_testing_settings, load_training_settings


def add_experience_to_memory(memory: Memory, history: list[Record]) -> None:
    for i in range(len(history) - 1):
        s = history[i].state
        a = history[i].action
        r = history[i].reward
        s_next = history[i + 1].state
        memory.add_sample((s, a, r, s_next))


def update_training_stats(
    episode_history: list[Record],
    env_stats: list[EnvStats],
    max_steps: int,
    training_stats: dict,
):
    # accumulate only negative rewards for clearer trend
    sum_neg_reward = sum([record.reward for record in episode_history if record.reward < 0])
    training_stats["sum_neg_reward"].append(sum_neg_reward)

    sum_queue_length = sum([el.queue_length for el in env_stats])
    training_stats["avg_queue_length"].append(round(sum_queue_length / max_steps, 1))

    # 1 car in queue for 1 step == 1 second of waiting time
    training_stats["cumulative_wait"].append(sum_queue_length)

    return training_stats


def training_session(settings_file: Path, out_path: Path):
    settings = load_training_settings(settings_file)

    memory = Memory(size_max=settings.memory_size_max, size_min=settings.memory_size_min)

    agent = Agent(settings=settings)

    episode = 0
    timestamp_start = datetime.now()
    tot_episodes = settings.total_episodes

    training_stats = {  # TODO use better struct
        "sum_neg_reward": [],
        "cumulative_wait": [],
        "avg_queue_length": [],
    }

    while episode < tot_episodes:
        print(f"\n----- Episode {episode + 1} of {tot_episodes}")

        new_epsilon = round(1.0 - (episode / tot_episodes), 2)
        agent.set_epsilon(new_epsilon)

        env = Environment(
            state_size=settings.state_size,
            n_cars_generated=settings.n_cars_generated,
            max_steps=settings.max_steps,
            yellow_duration=settings.yellow_duration,
            green_duration=settings.green_duration,
            gui=settings.gui,
            sumocfg_file=settings.sumocfg_file,
        )

        episode_history, env_stats = run_episode(env=env, agent=agent, seed=episode)

        add_experience_to_memory(memory=memory, history=episode_history)

        for _ in range(settings.training_epochs):
            agent.replay(memory=memory, gamma=settings.gamma)

        training_stats = update_training_stats(
            episode_history=episode_history,
            env_stats=env_stats,
            max_steps=settings.max_steps,
            training_stats=training_stats,
        )

        print(f"Epsilon: {agent.epsilon}")
        print(f"Reward: {training_stats['sum_neg_reward'][episode]}")
        print(f"Cumulative wait: {training_stats['cumulative_wait'][episode]}")
        print(f"Avg queue: {training_stats['avg_queue_length'][episode]}")

        episode += 1

    out_path.mkdir(parents=True, exist_ok=True)
    agent.save_model(out_path)

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.now())
    print("----- Session info saved at:", out_path)

    copyfile(src=settings_file, dst=out_path / TRAINING_SETTINGS_FILE)

    save_data_and_plot(
        data=training_stats["sum_neg_reward"],
        filename="reward",
        xlabel="Episode",
        ylabel="Cumulative negative reward",
        out_folder=out_path,
    )
    save_data_and_plot(
        data=training_stats["cumulative_wait"],
        filename="delay",
        xlabel="Episode",
        ylabel="Cumulative delay (s)",
        out_folder=out_path,
    )
    save_data_and_plot(
        data=training_stats["avg_queue_length"],
        filename="queue",
        xlabel="Episode",
        ylabel="Average queue length (vehicles)",
        out_folder=out_path,
    )


def testing_session(settings_file: Path, model_path: Path) -> None:
    """Load a trained agent from model_path and run a single testing episode."""
    settings = load_testing_settings(settings_file)

    agent = Agent(
        settings=load_training_settings(model_path / TRAINING_SETTINGS_FILE),
        epsilon=0,
        model_path=model_path,
    )

    env = Environment(
        state_size=settings.state_size,
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        gui=settings.gui,
        sumocfg_file=settings.sumocfg_file,
    )

    episode_history, env_stats = run_episode(env=env, agent=agent, seed=settings.episode_seed)
    testing_stats = {"reward": [], "queue_length": []}  # TODO use better struct

    for record in episode_history:
        testing_stats["reward"].append(record.reward)

    for stats in env_stats:
        testing_stats["queue_length"].append(stats.queue_length)

    # TODO enable multiple tests, let user define name, check for existenve ena overwrite
    test_path = model_path / "test"
    test_path.mkdir(parents=True, exist_ok=True)

    copyfile(src=settings_file, dst=test_path / TESTING_SETTINGS_FILE)

    save_data_and_plot(
        data=testing_stats["reward"],
        filename="reward",
        xlabel="Action step",
        ylabel="Reward",
        out_folder=test_path,
    )
    save_data_and_plot(
        data=testing_stats["queue_length"],
        filename="queue",
        xlabel="Step",
        ylabel="Queue length (vehicles)",
        out_folder=test_path,
    )

    print("----- Testing results saved at:", test_path)
