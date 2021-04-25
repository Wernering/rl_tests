# Reinforcement Learning An Introduction Page 130. Windy Gridworld. Use of SARSA

from random import random, choice
import time
import os

from game_config import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.set_printoptions(precision=1)


class Player:

    def __init__(self, gw_rows, gw_cols, card_mov=4):
        """
        'Character' or player that moves in the GridWorld

        :param card_mov: cardinality of movements (4, 8, 9)
        """

        # GridWorld Limitations
        self.gw_rows, self.gw_cols = gw_rows, gw_cols

        # Current position
        self.cy, self.cx = None, None

        movesets = {
            4: [self.move_up, self.move_down, self.move_right, self.move_left],
            8: [self.move_up, self.move_down, self.move_right, self.move_left,
                self.move_up_right, self.move_up_left, self.move_down_right, self.move_down_left],
            9: [self.move_up, self.move_down, self.move_right, self.move_left,
                self.move_up_right, self.move_up_left, self.move_down_right, self.move_down_left,
                self.not_move],
        }

        # TODO: change the option of movesets acording to a confing of available/not available
        self.available_moveset = movesets[card_mov]

        # Number of movements
        self.movements_count = 0

    def add_movement_count(self):
        self.movements_count += 1

    def reset_movements_count(self):
        self.movements_count = 0

    def set_position(self, y, x):
        self.cy, self.cx = y, x

    # Basic Movements (Cardinality 4)
    def move_up(self, y, x):
        return max(y - 1, 0), x

    def move_down(self, y, x):
        return min(y + 1, self.gw_rows), x

    def move_left(self, y, x):
        return y, max(x - 1, 0)

    def move_right(self, y, x):
        return y, min(x + 1, self.gw_cols)

    # King Movements (Cardinality 8)
    def move_up_right(self, y, x):
        aux1 = self.move_up(y, x)
        aux2 = self.move_right(y, x)
        return aux2[0], aux1[1]

    def move_up_left(self, y, x):
        aux1 = self.move_up(y, x)
        aux2 = self.move_left(y, x)
        return aux1[0], aux2[1]

    def move_down_right(self, y, x):
        aux1 = self.move_down(y, x)
        aux2 = self.move_right(y, x)
        return aux1[0], aux2[1]

    def move_down_left(self, y, x):
        aux1 = self.move_down(y, x)
        aux2 = self.move_left(y, x)
        return aux1[0], aux2[1]

    # Do Nothing (Cardinality 9)
    def not_move(self, y, x):
        return y, x


class GridWorld:

    def __init__(self, rows, columns, wind=None):
        """"""

        self.rows = rows
        self.columns = columns

        # Wind matrix
        self.wind_matrix = np.zeros((2, self.rows + 1, self.columns + 1)).astype(int)

        self.configure_wind_matrix(wind)

    def afterwind_pos(self, y_init, x_init):
        """

        :param x_init: x position of player after its movement
        :param y_init: y position of player after its movement

        :return: tuple (x, y) Position of player after wind effect
        """

        aw_x = x_init + self.wind_matrix[0, y_init, x_init]
        aw_y = y_init + self.wind_matrix[1, y_init, x_init]

        # Keep player inside x-direction
        if aw_x > self.columns:
            aw_x = self.columns

        elif aw_x < 0:
            aw_x = 0

        # Keep player inside y-direction
        if aw_y > self.rows:
            aw_y = self.rows

        elif aw_y < 0:
            aw_y = 0

        return aw_y, aw_x

    def configure_wind_matrix(self, wind_config):
        """
        Configures the wind matrix of the gridworld

        :param wind_config:
        :return:
        """
        if isinstance(wind_config, dict):
            if "rows" in wind_config:
                for k, v in wind_config["rows"].items():
                    self.wind_matrix[0, k, :] = v

            if "columns" in wind_config:
                for k, v in wind_config["columns"].items():
                    self.wind_matrix[1, :, k] = v


class Agent(Player):
    def __init__(self, g_rows, g_cols, alpha=0.5, epsilon=0.1, evt1="SARSA", card_mov=4):
        """
        SubClass of Player. Controlled by and algorithm

        :param g_rows: number of gridworld rows
        :param g_cols: number of gridworld columns
        :param alpha: learning rate
        :param epsilon: epsilon value for random action
        :param evt1: method to calculate Expected Value t1. SARSA, EX-SARSA, Q-LEARN
        """

        super().__init__(g_rows, g_cols, card_mov=card_mov)

        self.alpha = alpha
        self.epsilon = epsilon

        # Method to get expected value in t1
        self.evt1 = evt1

        # Create state-action matrix. Only 0's as initial values
        self.sa_matrix = np.zeros((len(self.available_moveset), g_rows + 1, g_cols + 1))

        # Define moveset with id (to locate to matrix)
        self.moveset = {i: m for i, m in enumerate(self.available_moveset)}

        # Routes settings. Current and optimal
        self.current_route = []
        self.optimal_route = []

    def update_epsilon(self, x):
        if 0 <= x < 1:
            self.epsilon = x

    def evaluate_next_move(self, id_move, f_row, f_col):
        """

        :param id_move: id of the movement of the player
        :param f_row: final row after movement + wind interaction
        :param f_col: final column after movement + wind interaction
        :return: expected value of the final movement
        """
        return self.sa_matrix[id_move, f_row, f_col]

    def select_next_move(self, evaluations):
        """
        :param evaluations: dictionary with id_move: exected_value
        :return: id_move it will make
        """

        # If random value is less than epsilon value, choose random movement
        if random() < self.epsilon:
            return choice([(i, v) for i, v in evaluations.items()])

        # Else, choose the movement with max expected value. If 2 are the same, choose randomly between them
        return choice([(i, v) for i, v in evaluations.items() if v == max(evaluations.values())])

    def return_position(self):
        return self.cy, self.cx

    def update_sa_matrix(self, id_move_t, row_t, col_t, reward_t1, expected_value_t1):
        self.sa_matrix[id_move_t, row_t, col_t] += self.alpha * (
                reward_t1 + expected_value_t1 - self.sa_matrix[id_move_t, row_t, col_t])

    def update_current_route(self, id_mov, y, x):
        self.current_route.append((id_mov, [y, x]))

    def reset_current_route(self):
        self.current_route = []

    def update_optimal_route(self):
        if not self.optimal_route:
            self.optimal_route = self.current_route

        else:
            if len(self.current_route) < len(self.optimal_route):
                self.optimal_route = self.current_route

    def evaluate_evt1(self, y, x, y_end, x_end):

        # If position is the same as endpoint, return 0 always.
        if (y, x) == (y_end, x_end):
            return 0

        if self.evt1 == "Q-LEARN":
            return max([self.sa_matrix[i, y, x] for i in self.moveset.keys()])

        if self.evt1 == "EX-SARSA":

            value = max([self.sa_matrix[i, y, x] for i in self.moveset.keys()]) * (1 - self.epsilon)
            value += sum(
                [self.sa_matrix[i, y, x] * self.epsilon / len(self.moveset.keys()) for i in self.moveset.keys()])

            return value

        else:  # self.evt1 == "SARSA"
            evaluations = {i: self.sa_matrix[i, y, x] for i in self.moveset.keys()}
            id_mov, value = self.select_next_move(evaluations)
            return value


class AIGame:

    def __init__(self, gw_rows, gw_cols, gw_wind, sp, ep, agent_alpha, agent_epsilon, agent_evt1, agent_card_mov):

        # Starting Point and endpoint
        self.sp = sp
        self.ep = ep

        # Create GridWorld
        self.GW = GridWorld(gw_rows, gw_cols, wind=gw_wind)

        # Create Agent
        self.Agent = Agent(gw_rows, gw_cols, alpha=agent_alpha, epsilon=agent_epsilon, evt1=agent_evt1,
                           card_mov=agent_card_mov)
        self.Agent.set_position(*self.sp)

        # Save Info
        self.total_iterations = 0
        self.total_cycles = 0
        self.cycle_iterations = 0

        # Dataframes of resuts
        self.df_iter_cycle = pd.DataFrame(columns=["iteration", "cycles"])
        self.df_cycle_steps = pd.DataFrame(columns=["cycle", "steps"])

    def return_reward(self, y, x):
        """
        Returns the reward of the current position of the agent.
        If the position is the same as the ending point the reward is 0, else is -1

        """
        if (y, x) == self.ep:
            return 0
        return -1

    def add_cycle(self):
        self.total_cycles += 1

    def add_iteration(self):
        self.total_iterations += 1
        self.cycle_iterations += 1

    def reset_iters(self):
        self.cycle_iterations = 0

    def check_victory(self):
        if self.ep == self.Agent.return_position():
            return True
        return False

    def save_iteration(self):
        self.df_iter_cycle = self.df_iter_cycle.append(
            pd.Series([self.total_iterations, self.total_cycles], index=self.df_iter_cycle.columns), ignore_index=True)

    def save_cycle(self):
        self.df_cycle_steps = self.df_cycle_steps.append(
            pd.Series([self.total_cycles, self.cycle_iterations], index=self.df_cycle_steps.columns), ignore_index=True)

    def movement_destination(self, id_mov, y, x):
        return self.GW.afterwind_pos(*self.Agent.moveset[id_mov](y, x))

    def turn(self, ui=False):
        """
        All the actions that are made in one turn

        :param ui: defines if the training will be seen in the game or not.
        """
        # Add iteration (so we start from 1)
        self.add_iteration()

        # Select movement
        evaluations = {i: self.Agent.sa_matrix[i, self.Agent.cy, self.Agent.cx] for i in self.Agent.moveset.keys()}
        id_mov, value = self.Agent.select_next_move(evaluations)

        # Get destination from action
        y_t1, x_t1 = self.movement_destination(id_mov, self.Agent.cy, self.Agent.cx)

        # Get reward
        reward = self.return_reward(y_t1, x_t1)

        # Obtain Expected value for t1
        expected_value_t1 = self.Agent.evaluate_evt1(y_t1, x_t1, *self.ep)

        # Update state-action matrix
        self.Agent.update_sa_matrix(id_move_t=id_mov, row_t=self.Agent.cy, col_t=self.Agent.cx, reward_t1=reward,
                                    expected_value_t1=expected_value_t1)

        # set new position of player
        self.Agent.set_position(y_t1, x_t1)

        # Append the action - state to the current route
        self.Agent.update_current_route(id_mov, y_t1, x_t1)

        # Check result
        if self.check_victory():

            if not ui:
                # if ui, the update to starting position is made afterwards, in the Game Loop
                self.Agent.set_position(*self.sp)

            self.add_cycle()
            self.save_cycle()

            # Update optimal route and reset current route
            self.Agent.update_optimal_route()
            self.Agent.reset_current_route()

            self.reset_iters()

        # Write info in dataframe
        self.save_iteration()

    def loop(self, turns=30):
        """
        All the actions that are made in between turns.

        :param turns: Amount of turns to play
        """
        for _ in range(turns):
            # Make a turn (all the steps above)
            self.turn()

    def graph(self, show=True, save=False):
        """
        :param show: if True Show graph generated
        :param save: if True, saves graph generated
        """

        def image():
            if self.df_cycle_steps.empty:
                fig, ax = plt.subplots(nrows=1, ncols=1)
                fig.suptitle(
                    f"{self.Agent.evt1}-{self.Agent.alpha}-{self.Agent.epsilon}-{len(self.Agent.moveset)} movements")
                self.df_iter_cycle["cycles"].plot(ax=ax)

            else:
                fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
                fig.suptitle(
                    f"{self.Agent.evt1}-{self.Agent.alpha}-{self.Agent.epsilon}-{len(self.Agent.moveset)} movements")

                self.df_iter_cycle["cycles"].plot(ax=ax1)
                self.df_cycle_steps["steps"].plot(ax=ax2)

        if show:
            image()
            plt.show()

        if save:
            if not os.path.isdir("Graficos"):
                os.mkdir("Graficos")

            image()
            plt.savefig(
                f"Graficos/{self.Agent.evt1}-a_{self.Agent.alpha}-e_{self.Agent.epsilon}-move_card_{len(self.Agent.available_moveset)}-turns_{self.total_iterations}.png")

    def return_optimal_rout(self):
        return len(self.Agent.optimal_route), self.Agent.optimal_route


if __name__ == "__main__":
    # Tiempo Inicio
    t_ini = time.time()

    # Create Object with game
    aig = AIGame(gw_rows=GRIDWORLD_ROWS,
                 gw_cols=GRIDWORLD_COLUMNS,
                 gw_wind=GRIDWORLD_WIND,
                 sp=STARTING_POINT,
                 ep=ENDING_POINT,
                 agent_alpha=ALPHA,
                 agent_epsilon=EPSILON,
                 agent_evt1=TD_METHOD,
                 agent_card_mov=CARDINALITY_MOVESET)

    # Loop Game through 8000 steps
    aig.loop(turns=8000)

    # Show Matrix
    print(aig.Agent.sa_matrix)

    # Print time of Execution
    print(f"Tiempo de ejecucion: {time.time() - t_ini}")

    # Graph results
    aig.graph(show=False, save=True)

    # Print optimal number of steps and route
    print(*aig.return_optimal_rout())
