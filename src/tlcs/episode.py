from dataclasses import dataclass

from numpy.typing import NDArray

from tlcs.agent import Agent
from tlcs.env import Environment, EnvStats


@dataclass
class Record:
    state: NDArray
    action: int
    reward: float


def run_episode(env: Environment, agent: Agent, seed: int) -> tuple[list[Record], list[EnvStats]]:
    env.generate_routefile(seed=seed)

    previous_total_wait = 0.0
    history: list[Record] = []
    env_stats: list[EnvStats] = []

    env.activate()

    while not env.is_over():
        current_state = env.get_state()
        action = agent.choose_action(current_state)
        action_stats = env.execute(action)

        env_stats += action_stats

        current_total_wait = env.get_cumulated_waiting_time()
        reward = previous_total_wait - current_total_wait

        previous_total_wait = current_total_wait

        history.append(Record(state=current_state, action=action, reward=reward))

    env.deactivate()

    return history, env_stats
