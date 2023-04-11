import logging

import numpy as np

from config.logger import LOG_NAME

logger = logging.getLogger(LOG_NAME)


class Maze:
    def __init__(
        self, rows: int, columns: int, walls: list[tuple[int, int]] = []
    ) -> None:
        self.rows = rows
        self.columns = columns
        self.walls = walls

        self.maze_shape = self.create_shape()
        logger.info(
            f"Maze shape created successfully. Shape {self.get_maze_dimension()}"
        )

    def create_shape(self):
        # Maze is created as a numpy array (y, x)
        maze = np.zeros((self.rows, self.columns))

        # WALLS is an array of matrix position (row, column) -> (y, x)
        for wall_y, wall_x in self.walls:
            if (0 <= wall_y <= self.rows) and (0 <= wall_x <= self.columns):
                maze[wall_y - 1, wall_x - 1] = 1

            else:
                logger.warning(
                    f"WALL ({wall_y}, {wall_x}) was not able to be positioned. Is going to be ignored"
                )
        return maze

    def get_maze_dimension(self):
        return self.maze_shape.shape
