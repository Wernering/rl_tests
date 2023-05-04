# Reinforcement Learning. An Introduction. Page 82
# Jack's Car Rental (Non-Lineal)

import logging
import datetime as dt
from logging.config import dictConfig

from classes.rental import JacksRental
from functions.execute import synchronous_solve
from functions.utils import graph_state_value, timer, graph_policy_value
from config.logger import LOG_NAME, LoggerConfig

dictConfig(LoggerConfig().dict())

logger = logging.getLogger(LOG_NAME)

max_moves = 5
move_cost = 2
lease_reward = 10

location_a = {
    "capacity": 20,
    "lease_lambda": 3,
    "return_lambda": 3,
    "epsilon": 0.001,
}

location_b = {
    "capacity": 20,
    "lease_lambda": 4,
    "return_lambda": 2,
    "epsilon": 0.001,
}

theta_division = 10

non_lineal = {
    "parking_cost": 4,
    "free_cars": 10,
    "free_a_to_b": 1,
}

problem = JacksRental(
    max_moves=max_moves,
    cost=move_cost,
    reward=lease_reward,
    location_a=location_a,
    location_b=location_b,
    **non_lineal
)


with timer("Complete Exercise"):
    solution = synchronous_solve(problem, theta_update=theta_division)

id = dt.date.today()
graph_state_value(solution, id)
graph_policy_value(solution, id)
