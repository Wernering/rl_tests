# Local
from ..config import LOGGER


class Node:
    def __init__(self, name: str, initial_v: float = 0.5):
        LOGGER.info(f"Creating Node {name}")
        self.name = name
        self.initial_v = initial_v
        self.V = initial_v

    def td_update(self, r: float, v_s1: float, alpha: float, gamma: float) -> None:
        """
        :param r: Inmediate Reward of action taken
        :param v_s1: Expected Reward of state S'
        """
        self.V = self.V + alpha * (r + gamma * v_s1 - self.V)

    def mc_update(self, Gt, alpha):
        """
        :param Gt: Final reward of the episode
        :param alpha: step size. Learning rate
        """
        self.V = self.V + alpha * (Gt - self.V)

    def restart_v(self) -> None:
        LOGGER.debug(f"Restarting Node {self.name}")
        self.V = self.initial_v
