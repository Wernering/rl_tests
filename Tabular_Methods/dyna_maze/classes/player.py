import logging

import numpy as np

from config.logger import LOG_NAME

logger = logging.getLogger(LOG_NAME)


class Player:
    def __init__(self, start: np.ndarray) -> None:
        self.current_position = np.array([start[0], start[1]])

    def update_position(self, new_position: np.ndarray):
        self.current_position = new_position
        logger.debug(
            f"Player's current position: {self.position_as_matrix(self.current_position)}"
        )

    def position_as_matrix(self, value: np.ndarray) -> np.ndarray:
        return value + [1, 1]

    def move_up(self, possible=True):
        if possible:
            return self.current_position + [-1, 0]
        return self.current_position

    def move_down(self, possible=True):
        if possible:
            return self.current_position + [1, 0]
        return self.current_position

    def move_right(self, possible=True):
        if possible:
            return self.current_position + [0, 1]
        return self.current_position

    def move_left(self, possible=True):
        if possible:
            return self.current_position + [0, -1]
        return self.current_position
