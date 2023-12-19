# Reinforcement Learning An Introduction. Page 164. Dyna Maze

# Standard Library
import logging
import time
from logging.config import dictConfig

# External
import config.config as config
from config.logger import LOG_NAME, LoggerConfig
from functions.training import cycle
from functions.utils import graph


dictConfig(LoggerConfig().dict())

logger = logging.getLogger(LOG_NAME)

if __name__ == "__main__":
    # RNG_SEED
    seed = config.RND_SEED

    # Positions
    start = config.START
    end = config.END

    # Maze
    rows = config.HEIGHT
    columns = config.WIDTH
    walls = config.WALLS

    # Learning
    alpha = config.ALPHA
    gamma = config.GAMMA
    epsilon = config.EPSILON
    episodes = config.EPISODES

    results = {}

    for planning_steps in [0, 5, 50]:
        logger.info(f"Start cycle with steps: {planning_steps}")

        t1 = time.time()
        results[planning_steps] = cycle(
            rows=rows,
            columns=columns,
            start=start,
            end=end,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            steps=planning_steps,
            walls=walls,
            seed=seed,
            episodes=episodes,
        )
        logger.info(f"Execution time: {time.time() - t1} seconds")

    graph(results, seed=seed, alpha=alpha, gamma=gamma, epsilon=epsilon)
