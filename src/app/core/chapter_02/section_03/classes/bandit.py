# External
import numpy as np


class StationaryBandit:
    def __init__(self, ini: float = 0):
        self._q_star = self.calculate_q_star()
        self.initial_value = ini
        self._Q_a = ini
        self.selections = 0

    @staticmethod
    def calculate_q_star():
        """
        Calulates de real reward of the Bandit, out of a normal distribution
        with mean 0 and variance 1.
        """
        return np.random.normal(0, 1)

    def get_Qa(self):
        """
        Returns Qa of the bandit
        """
        return self._Q_a

    def yield_reward(self):
        """
        Calculates the reward yielded if the Bandit is selected.
        """
        return np.random.normal(self._q_star, 1)

    def increased_selected_times(self):
        """
        Updates the number of times this bandit has been selected.
        """
        self.selections += 1

    def restart(self):
        """
        Resets the variables for the next round.
        """

        self.selections = 0
        self._Q_a = self.initial_value

    def recalculate_estimated_reward(self, reward):
        """
        Recalculate the estimated reward according to the last yielded reward,
        as an average.
        """
        self._Q_a = self._Q_a + (1 / self.selections) * (reward - self._Q_a)

    def selected(self):
        """
        Steps executed if the bandit is selected. Update variables for
        the next iteration.
        """
        reward = self.yield_reward()

        self.increased_selected_times()
        if self.selections == 1:
            self._Q_a = reward
        else:
            self.recalculate_estimated_reward(reward)

        return reward
