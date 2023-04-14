# Reinforcement Learning. An Introduction. Page 28
# K-Armed-Bandit

from logging.config import dictConfig
import logging

from config.logger import LOG_NAME, LoggerConfig
from functions.utils import graph
from functions.execute import execute

dictConfig(LoggerConfig().dict())

logger = logging.getLogger(LOG_NAME)

epsilons = [0.01, 0.1, 0, 0.5]
k = 10
episodes = 1000
cycles = 2000

experiment_results = execute(epsilons=epsilons, k=k, episodes=episodes, cycles=cycles)

graph(experiment_results, episodes=episodes, cycles=cycles)