import platform

PATH = "./k_armed_bandit/"
GRAPH_PATH = f"{PATH}/graphs"
LOG_PATH = f"{PATH}/info.log"

if platform.system() == "Windows":
    PATH = ".\\k_armed_bandit"
    GRAPH_PATH = f"{PATH}\\graphs"
    LOG_PATH = f"{PATH}\\info.log"
