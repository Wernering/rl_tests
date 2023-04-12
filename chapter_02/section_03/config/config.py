import platform

PATH = "./chapter_02/section_03"
GRAPH_PATH = f"{PATH}/graphs/"
LOG_PATH = f"{PATH}/info.log"

if platform.system() == "Windows":
    PATH = ".\\chapter_02\\section_03"
    GRAPH_PATH = f"{PATH}\\graphs\\"
    LOG_PATH = f"{PATH}\\info.log"
