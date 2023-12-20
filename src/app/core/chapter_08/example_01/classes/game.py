# External
import numpy as np

# Local
from ..config import LOGGER
from ..functions.utils import get_value_as_matrix
from .maze import Maze
from .player import Player


class Game:
    def __init__(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        rows: int,
        columns: int,
        walls: list[tuple[int, int]] = [],
        **kw,
    ) -> None:
        self.win_reward = kw.get("win_reward", 1)
        self.no_win_reward = kw.get("no_win_reward", 0)

        self.start_position = np.array((start[0] - 1, start[1] - 1))
        self.end_position = np.array((end[0] - 1, end[1] - 1))

        self.maze = Maze(rows=rows, columns=columns, walls=walls)
        self.rows, self.columns = self.maze.get_maze_dimension()
        self.player = Player(start=self.start_position)

        LOGGER.info(
            f"Starting Position: {get_value_as_matrix(self.start_position)}. Ending Position:"
            f" {get_value_as_matrix(self.end_position)}"
        )

    def step(self, action: int) -> None:
        new_pos = self.player.current_position

        if action == 0:
            probe_position = self.player.current_position + [-1, 0]
            LOGGER.debug(f"Action: 'up'. Probe position: {get_value_as_matrix(probe_position)}")
            possible = False
            if probe_position[0] >= 0:
                possible = self.maze.maze_shape[*probe_position] == 0
            new_pos = self.player.move_up(possible=possible)

        elif action == 1:
            probe_position = self.player.current_position + [1, 0]
            LOGGER.debug(f"Action: 'down'. Probe position: {get_value_as_matrix(probe_position)}")
            possible = False
            if probe_position[0] < (self.rows):
                possible = self.maze.maze_shape[*probe_position] == 0
            new_pos = self.player.move_down(possible=possible)

        elif action == 2:
            probe_position = self.player.current_position + [0, -1]
            LOGGER.debug(f"Action: 'left'. Probe position: {get_value_as_matrix(probe_position)}")
            possible = False
            if probe_position[1] >= 0:
                possible = self.maze.maze_shape[*probe_position] == 0
            new_pos = self.player.move_left(possible=possible)

        elif action == 3:
            probe_position = self.player.current_position + [0, 1]
            LOGGER.debug(f"Action: 'right'. Probe position: {get_value_as_matrix(probe_position)}")
            possible = False
            if probe_position[1] < (self.columns):
                possible = self.maze.maze_shape[*probe_position] == 0
            new_pos = self.player.move_right(possible=possible)

        self.player.update_position(new_pos)

    def win(self) -> bool:
        if (self.player.current_position == self.end_position).all():
            return True
        return False

    def reward(self) -> int:
        if self.win():
            return self.win_reward
        return self.no_win_reward

    def start(self) -> None:
        self.player.update_position(self.start_position)

    def get_player_current_position(self) -> np.ndarray:
        return self.player.current_position

    def print_current_situation(self) -> str:
        string = ""
        for row in range(self.maze.rows):
            string += "\n"
            for col in range(self.maze.columns):
                position = np.array([row, col])
                if np.all(position == self.player.current_position):
                    string += " p"

                elif self.maze.maze_shape[row, col] == 1:
                    string += " x"

                elif np.all(position == self.start_position):
                    string += " s"

                elif np.all(position == self.end_position):
                    string += " e"

                else:
                    string += " o"

        return string
