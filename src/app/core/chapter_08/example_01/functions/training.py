# Standard Library
import logging
import random

# External
from classes.agent import Agent
from classes.game import Game
from config.logger import LOG_NAME

# Local
from .utils import get_value_as_matrix


logger = logging.getLogger(LOG_NAME)


def episode(agent: Agent, game: Game) -> tuple[Agent, int]:
    game.start()
    step = 0
    while not game.win():
        # while step < 10:
        step += 1

        # get step's initial-State S
        state_s = game.get_player_current_position()

        # Action taken acording to agent policy
        action = agent.take_action(state_s)

        # Action done in game
        game.step(action)

        # Get game reward for previous step.
        reward = game.reward()

        # Get step's end-State S'
        state_s1 = game.get_player_current_position()

        # Transform states to tuple
        state_s1 = tuple(state_s1)
        state_s = tuple(state_s)

        logger.debug(
            f"Step :{step}. Agent's Start Position: {get_value_as_matrix(state_s)}. Action taken:"
            f" {action}. Agent's End position: {get_value_as_matrix(state_s1)}. Reward: {reward}."
            f" Expected value: {agent.expected_value[action, *state_s]}"
        )

        # Update expected value matrix
        agent.update_expected_value(state_s=state_s, action=action, reward=reward, state_s1=state_s1)

        # Update Agent Model
        agent.update_model(state_s=state_s, action=action, reward=reward, state_s1=state_s1)

        # Agent Model Learning. n = steps.
        agent.model_learning()

    return agent, step


def cycle(
    rows: int,
    columns: int,
    start: tuple,
    end: tuple,
    alpha: float,
    gamma: float,
    epsilon: float,
    steps: int,
    walls: list[tuple] = [],
    seed=None,
    episodes=1,
) -> dict:
    random.seed(seed)
    game = Game(
        start=start,
        end=end,
        rows=rows,
        columns=columns,
        walls=walls,
    )

    agent = Agent(
        rows=rows,
        columns=columns,
        movements=4,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        steps=steps,
    )

    results = {}

    for i in range(1, episodes + 1):
        agent, steps_taken = episode(agent, game)
        results[i] = steps_taken

    return results
