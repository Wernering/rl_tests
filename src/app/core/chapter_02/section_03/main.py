# Reinforcement Learning. An Introduction. Page 28
# K-Armed-Bandit

# Local
from .config import LOGGER
from .functions import execute, graph


def play():
    epsilons = [0.01, 0.1, 0, 0.5]
    k = 10
    episodes = 1_000
    cycles = 2_000

    LOGGER.info("Starting Calculation...")
    experiment_results = execute(epsilons=epsilons, k=k, episodes=episodes, cycles=cycles)
    graph(experiment_results, episodes=episodes, cycles=cycles)
