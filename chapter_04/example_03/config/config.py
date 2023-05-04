import platform

STATES = 100

PATH = "./chapter_04/example_03"
GRAPH_PATH = f"{PATH}/graphs"
LOG_PATH = f"{PATH}/info.log"

if platform.system() == "Windows":
    PATH = ".\\chapter_04\\example_03"
    GRAPH_PATH = f"{PATH}\\graphs"
    LOG_PATH = f"{PATH}\\info.log"
