# Reinforcement Learning. An Introduction. Page 81
# Jack's Car Rental

# Standard Library
import datetime as dt
import logging
from logging.config import dictConfig

# External
from classes.rental import JacksRental
from config.logger import LOG_NAME, LoggerConfig
from functions.execute import synchronous_solve
from functions.utils import graph_policy_value, graph_state_value, timer


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

problem = JacksRental(
    max_moves=max_moves,
    cost=move_cost,
    reward=lease_reward,
    location_a=location_a,
    location_b=location_b,
)


with timer("Complete Exercise"):
    solution = synchronous_solve(problem, theta_update=theta_division)

id = dt.date.today()
graph_state_value(solution, id)
graph_policy_value(solution, id)
