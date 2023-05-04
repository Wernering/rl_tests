import platform

PATH = "./chapter_04/excercise_07"
GRAPH_PATH = f"{PATH}/graphs"
LOG_PATH = f"{PATH}/info.log"

if platform.system() == "Windows":
    PATH = ".\\chapter_04\\excercise_07"
    GRAPH_PATH = f"{PATH}\\graphs"
    LOG_PATH = f"{PATH}\\info.log"
