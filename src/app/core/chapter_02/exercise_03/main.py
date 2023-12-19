# Reinforcement Learning. An Introduction. Page 33
# K-Armed-Bandit

# Local
from .config import LOGGER
from .functions import execute, graph


epsilons = [0.01, 0.1, 0, 0.5]
k = 10
episodes = 10_000
cycles = 2_000
alpha = 0.1


def play():
    LOGGER.info("Starting Calculation...")
    experiment_results = execute(epsilons=epsilons, k=k, episodes=episodes, cycles=cycles, alpha=alpha)
    graph(experiment_results, episodes=episodes, cycles=cycles, alpha=alpha)
