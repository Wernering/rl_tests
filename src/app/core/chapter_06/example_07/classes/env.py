# Standard Library
from typing import Callable

# Local
from ..config import LOGGER


class Node:
    def __init__(self, name: str) -> None:
        LOGGER.info(f"Creating Node: {name}")
        self.name: str = name
        self.number_actions = 0
        self.available_actions: dict[int, tuple[Node, Callable]] = {}

    def create_action(self, dest_node: "Node", reward_func: Callable) -> None:
        LOGGER.info(f"Adding action from Node {self.name} to Node {dest_node}")
        self.available_actions[self.number_actions] = dest_node, reward_func
        self.number_actions += 1

    def actions(self) -> dict:
        return self.available_actions

    def __repr__(self) -> str:
        return self.name

    def __str__(self):
        return self.name


class Env:
    def __init__(self, config: list[tuple[str, tuple[str, Callable]]], start_node: str, end_nodes: list[str]) -> None:
        LOGGER.info("Initializing Environment with specified configuration")

        self.nodes: dict[str, Node] = {}

        for node, val in config:
            dest_node, fun = val

            if node not in self.nodes:
                self.nodes[node] = Node(name=node)

            if dest_node not in self.nodes:
                self.nodes[dest_node] = Node(name=dest_node)

            node = self.nodes.get(node)
            dest_node = self.nodes.get(dest_node)

            node.create_action(dest_node, reward_func=fun)

        self.start_node: Node = self.nodes.get(start_node)
        self.state: Node = self.start_node

        self.end_nodes = [n for k, n in self.nodes.items() if k in end_nodes]

    def reset(self) -> None:
        self.state = self.start_node

    def step(self, action: int) -> tuple:
        dest_node, reward_fun = self.state.available_actions.get(action)
        self.state = dest_node

        return self.end(), self.state, reward_fun()

    def end(self) -> bool:
        if self.state in self.end_nodes:
            return True
        return False
