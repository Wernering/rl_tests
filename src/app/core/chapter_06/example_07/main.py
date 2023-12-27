# Reinforcement Learning An Introduction Page 134. Double Learning

# External
import numpy as np

# Project
from app.utils import ctx_timer

# Local
from .classes import Agent, Env, Game, UpdateMethod
from .config import LOGGER
from .functions import graph


def play():
    RUNS = 100
    EPISODES = 300

    def reward_0():
        return 0

    def reward_rnd():
        return np.random.normal(loc=-0.1, scale=1)

    config = [
        ("A", ("B", reward_0)),
        ("A", ("Terminal_Right", reward_0)),
        ("B", ("Terminal_Left", reward_rnd)),
        ("B", ("Terminal_Left", reward_rnd)),
        ("B", ("Terminal_Left", reward_rnd)),
        ("B", ("Terminal_Left", reward_rnd)),
        ("B", ("Terminal_Left", reward_rnd)),
        ("B", ("Terminal_Left", reward_rnd)),
        ("B", ("Terminal_Left", reward_rnd)),
        ("B", ("Terminal_Left", reward_rnd)),
        ("B", ("Terminal_Left", reward_rnd)),
        ("B", ("Terminal_Left", reward_rnd)),
        ("Terminal_Left", ("Terminal_Left", reward_0)),
        ("Terminal_Right", ("Terminal_Right", reward_0)),
    ]

    env = Env(config=config, start_node="A", end_nodes=["Terminal_Left", "Terminal_Right"])

    results = []

    with ctx_timer(f"Whole Excercise", logger=LOGGER):
        for agent in (
            Agent(nodes=env.nodes.values(), update_method=UpdateMethod.QL),
            Agent(nodes=env.nodes.values(), update_method=UpdateMethod.DQL),
        ):
            with ctx_timer(f"Agent-> {agent.update_method.value}", logger=LOGGER):
                game = Game(agent=agent, env=env)

                game.run(terminal_state="Terminal_Left", runs=RUNS, episodes=EPISODES)

                game.game_avg()

                results.append((agent.update_method.value, game.avg))

    graph(results, episodes=EPISODES, cycles=RUNS)
