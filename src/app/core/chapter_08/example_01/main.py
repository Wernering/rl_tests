# Reinforcement Learning An Introduction. Page 164. Dyna Maze

# Local
from .config import LOGGER, config
from .functions.training import cycle
from .functions.utils import graph


def play():
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
        LOGGER.info(f"Start cycle with steps: {planning_steps}")

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

    graph(results, seed=seed, alpha=alpha, gamma=gamma, epsilon=epsilon)
