# Standard Library
import copy

# External
import numpy as np

# Local
from ..config import LOGGER
from .agent import Agent
from .env import Env


class Game:
    def __init__(self, agent: Agent, env: Env) -> None:
        self.agent = agent
        self.env = env

        self.result = None
        self.avg = None

    def game_avg(self) -> None:
        self.avg = np.mean(self.result, axis=0)

    def time_step(self):
        initial_state = copy.copy(self.env.state)
        action = self.agent.choose_action(initial_state)
        end, end_state, reward = self.env.step(action)
        self.agent.update_q_values(initial_state=initial_state, action=action, end_state=end_state, reward=reward)
        return end

    def episode(self):
        end = False
        while not end:
            end = self.time_step()

    def cycle(self, terminal_state: str, episodes=300) -> None:
        list_actions = []
        for _ in range(episodes):
            self.env.reset()
            self.episode()
            list_actions.append(self.env.state.name == terminal_state)
        return list_actions

    def run(self, terminal_state: str, runs=10000, episodes=300) -> None:
        results = []
        for i in range(1, runs + 1):
            self.agent.restart()
            results.append(self.cycle(terminal_state=terminal_state, episodes=episodes))
            LOGGER.debug(f"Cycle {i} Done")
        self.result = np.array(results)
