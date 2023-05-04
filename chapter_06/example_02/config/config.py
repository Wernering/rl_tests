import platform

PATH = "./chapter_06/example_02"
GRAPH_PATH = f"{PATH}/graphs"
LOG_PATH = f"{PATH}/info.log"

if platform.system() == "Windows":
    PATH = ".\\chapter_06\\example_02"
    GRAPH_PATH = f"{PATH}\\graphs"
    LOG_PATH = f"{PATH}\\info.log"
