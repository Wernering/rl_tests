import numpy as np


class NonStationaryBandit:
    def __init__(self, alpha: float = 0.1, ini: float = 0):
        self._q_star = self.calculate_q_star()
        self.original_q_star = self._q_star
        self.alpha = alpha

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
        self._q_star = self.original_q_star

    def recalculate_estimated_reward(self, reward):
        """
        Recalculate the estimated reward according to the last yielded reward,
        acording to the step-size alpha
        """
        self._Q_a += self.alpha * (reward - self._Q_a)

    def rw_q_star(self):
        """
        Modify the real reward of the Bandit as a random walk with Normal distribution
        (mean 0, variance 0.0001)
        """
        self._q_star += np.random.normal(0, 0.01)

    def selected(self):
        """
        Steps executed if the bandit is selected. Update variables for the
        next iteration.
        """

        reward = self.yield_reward()

        self.increased_selected_times()
        self.recalculate_estimated_reward(reward)

        return reward
