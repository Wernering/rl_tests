
# Gridworld Configs
GRIDWORLD_ROWS = 6
GRIDWORLD_COLUMNS = 9

# Start - End
STARTING_POINT = (3, 0)  # (row , column) Starting from 0
ENDING_POINT = (3, 7)  # (row, column) starting from 0

# Agent configs
CARDINALITY_MOVESET = 4  # 4, 8, 9
ALPHA = 0.5
EPSILON = 0.1
TD_METHOD = "SARSA"  # SARSA, EX-SARSA, Q-LEARN

# Wind Instructions
GRIDWORLD_WIND = {
    "columns": {
        3: -1,
        4: -1,
        5: -1,
        6: -2,
        7: -2,
        8: -1
    },
    "rows": {
        # 1: -1,
        # 3: 2,
        # 6: -2
    }
}

# Pygame variables:
WINDOW_HEIGHT = 900
WINDOW_WIDTH = 1020
GRID_HEIGHT = 800
GRID_WIDTH = 1000
BORDER = 10

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

GREEN = (0, 102, 0)
ORANGE = (244, 153, 51)

RED = (255, 0, 0)
BLUE = (7, 17, 134)

MOVESET_POSITION = {
    0: "midtop",
    1: "midbottom",
    2: "midright",
    3: "midleft",
    4: "topright",
    5: "topleft",
    6: "bottomright",
    7: "bottomleft",
    8: "center"
}

POSITION_FIX = {
    0: (1, 1.01),
    1: (1, 0.99),
    2: (0.99, 1),
    3: (1.01, 1),
    4: (0.99, 1.01),
    5: (1.01, 1.01),
    6: (0.99, 0.99),
    7: (1.01, 0.99),
    8: (1, 1)
}