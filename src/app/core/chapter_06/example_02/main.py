# Standard Library
import datetime as dt
import logging
from logging.config import dictConfig

# External
from classes.model import MCModel, TDModel
from config.logger import LOG_NAME, LoggerConfig
from functions.utils import plot_estimated_value, plot_rms, timer


dictConfig(LoggerConfig().dict())

logger = logging.getLogger(LOG_NAME)

# Parameters
nodes = 5
real_probabilities = [x / (nodes + 1) for x in range(1, nodes + 1)]

gamma = 1
td_alphas = [0.15, 0.1, 0.05]
mc_alphas = [0.01, 0.02, 0.03, 0.04]

episodes = 100
runs = 100


id = dt.date.today()
rms = {}
for alpha in td_alphas:
    with timer(f"TD Model. Alpha: {alpha}"):
        td_model = TDModel(nodes=nodes, alpha=alpha, gamma=gamma)
        td_result = td_model.execute_many_runs(runs=runs, episodes=episodes)
        rms[f"TD-{alpha}"] = td_model.rms(real_probabilities)

for alpha in mc_alphas:
    with timer(f"MC Model. Alpha: {alpha}"):
        mc_model = MCModel(nodes=nodes, alpha=alpha, gamma=gamma)
        mc_result = mc_model.execute_many_runs(runs=runs, episodes=episodes)
        rms[f"MC-{alpha}"] = mc_model.rms(real_probabilities)


subset = {x: td_result[1][x] for x in [1, 2, 10, 100]}
subset["tv"] = real_probabilities
plot_estimated_value(subset, "TD_Estimated_Values", id)
plot_rms(rms, "RMS_Values", id)
