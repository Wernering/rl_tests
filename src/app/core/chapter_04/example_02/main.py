# Reinforcement Learning. An Introduction. Page 84
# Gambler's Problem

# Standard Library
import datetime as dt

# Project
from app.utils import ctx_timer

# Local
from .classes.problem import GamblersProblem
from .config import LOGGER, STATES
from .functions.execute import execute
from .functions.utils import graph_policy_value, graph_state_value


def play():
    head_probability = 0.4
    theta = 0.01

    problem = GamblersProblem(states=STATES, ph=head_probability)

    with ctx_timer("Whole Problem", logger=LOGGER):
        solution = execute(problem, theta=theta)

    id = dt.date.today()
    graph_state_value(solution, id, theta)
    graph_policy_value(solution, id, theta)
