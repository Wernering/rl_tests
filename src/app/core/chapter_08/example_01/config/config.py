# Standard Library
import platform


# Values considered as a matrix starting in (1,1).
# The transformation to numpy position is done inside the code

# Maze Parameters
HEIGHT = 6
WIDTH = 9
WALLS = [(2, 3), (3, 3), (4, 3), (5, 6), (1, 8), (2, 8), (3, 8)]

# Game Parameters
START = (3, 1)
END = (1, 9)

# # # # # # # # # # # #
#                     #
#  o o o o o o o x E  #
#  o o x o o o o x o  #
#  S o x o o o o x o  #
#  o o x o o o o o o  #
#  o o o o o x o o o  #
#  o o o o o o o o o  #
#                     #
# # # # # # # # # # # #

# Learning parameters
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.1

RND_SEED = 2.918
PLANNING_STEPS = 50
EPISODES = 50

PATH = "./chapter_08/example_01"
GRAPH_PATH = f"{PATH}/graphs"
LOG_PATH = f"{PATH}/info.log"

if platform.system() == "Windows":
    PATH = ".\\chapter_08\\example_01"
    GRAPH_PATH = f"{PATH}\\graphs"
    LOG_PATH = f"{PATH}\\info.log"
