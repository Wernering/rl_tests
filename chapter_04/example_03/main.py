import logging
from logging.config import dictConfig
import datetime as dt

from classes.problem import GamblersProblem
from functions.execute import execute
from functions.utils import graph_policy_value, graph_state_value, timer
from config.logger import LOG_NAME, LoggerConfig
import config.config as config

dictConfig(LoggerConfig().dict())

logger = logging.getLogger(LOG_NAME)

head_probability = 0.4
theta = 0.01

problem = GamblersProblem(states=config.STATES, ph=head_probability)

with timer("Whole Problem"):
    solution = execute(problem, theta=theta)

id = dt.datetime.now()
graph_state_value(solution, id, theta)
graph_policy_value(solution, id, theta)
