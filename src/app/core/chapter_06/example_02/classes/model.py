# Standard Library
import math
import random
import statistics

# Local
from ..config import LOGGER
from .node import Node


class Model:
    def __init__(
        self,
        nodes: int,
        alpha: float,
        gamma: float = 1,
        initial_node: str | None = None,
        initial_v: float = 0.5,
    ):
        self.number_nodes = nodes
        self.alpha = alpha
        self.gamma = gamma
        self.initial_v = initial_v

        self.model_list: list[Node] = self.initial_model()

        if initial_node is None:
            self.pos = int(nodes / 2)
        else:
            self.pos = int(initial_node)

        self.initial_position = self.pos
        self.results = {}

    def initial_model(self) -> list[Node]:
        model = []
        for i in range(self.number_nodes):
            model.append(Node(i, initial_v=self.initial_v))
        return model

    def step(self) -> int:
        if random.random() >= 0.5:
            return 1
        return -1

    def restart(self) -> None:
        for node in self.model_list:
            node.restart_v()

    def restart_initial_pos(self) -> None:
        self.pos = self.initial_position

    def execute_episode(self):
        pass

    def execute_run(self, episodes: int):
        LOGGER.debug("Starting run")
        episode_result = {}
        for episode in range(1, episodes + 1):
            self.execute_episode()
            episode_result[episode] = [x.V for x in self.model_list]
            self.restart_initial_pos()
        return episode_result

    def execute_many_runs(self, runs: int, episodes: int) -> dict:
        for run in range(1, runs + 1):
            self.restart()
            self.results[run] = self.execute_run(episodes=episodes)

        return self.results

    def rms(self, real_values: list) -> list:
        total_result = []
        for _, run_result in self.results.items():
            avg_result = []
            for ep_result in run_result.values():
                error = [(a - b) ** 2 for a, b in zip(ep_result, real_values)]
                avg_result.append(math.sqrt(sum(error) / len(error)))
            total_result.append(avg_result)
        return [statistics.mean(x) for x in zip(*total_result)]


class TDModel(Model):
    def __init__(
        self,
        nodes: int,
        alpha: float,
        gamma: float = 1,
        initial_node: str | None = None,
        initial_v: float = 0.5,
    ):
        super().__init__(nodes, alpha, gamma, initial_node, initial_v)

    def update_node(self, node: Node, r, vs):
        node.td_update(r, vs, self.alpha, self.gamma)

    def execute_episode(self):
        end = False
        while not end:
            action = self.step()

            # If we are on the further left node and we move to the left (-1)
            if self.pos == 0 and action == -1:
                r = 0
                v_s1 = 0
                end = True

            # If we are on the further right node and we move to the right (+1)
            elif (self.pos == self.number_nodes - 1) and action == 1:
                r = 1
                v_s1 = 0
                end = True

            else:
                v_s1: Node = self.model_list[self.pos + action].V
                r = 0

            self.update_node(self.model_list[self.pos], r, v_s1)
            self.pos += action


class MCModel(Model):
    def __init__(
        self,
        nodes: int,
        alpha: float,
        gamma: float = 1,
        initial_node: str | None = None,
        initial_v: float = 0.5,
    ):
        super().__init__(nodes, alpha, gamma, initial_node, initial_v)

        self.results = {}

    def update_node(self, node: Node, Gt: float):
        node.mc_update(Gt, self.alpha)

    def execute_episode(self):
        nodes = set()
        end = False

        while not end:
            # Add current node to the set of nodes of the episode
            nodes.add(self.model_list[self.pos])

            # Take an action
            action = self.step()

            # If we are on the further left node and we move to the left (-1)
            if self.pos == 0 and action == -1:
                r = 0
                end = True

            # If we are on the further right node and we move to the right (+1)
            elif (self.pos == self.number_nodes - 1) and action == 1:
                r = 1
                end = True

            self.pos += action

            if end:
                for node in nodes:
                    self.update_node(node, r)
