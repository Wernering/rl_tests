import numpy as np


class Bandit:
    def __init__(self, ini=0):

        self.q_star = self.calculate_q_star()
        self.ini = ini
        self.Q_a = ini
        self.n = 0

    @staticmethod
    def calculate_q_star():
        """
        Calulates de real reward of the Bandit, out of a normal distribution with mean 0 and variance 1.
        """
        return np.random.normal(0, 1)

    def yield_reward(self):
        """
        Calculates the reward yielded if the Bandit is selected.
        """
        return np.random.normal(self.q_star, 1)

    def increased_selected_times(self):
        """
        Updates the number of times this bandit has been selected.
        """
        self.n += 1

    def restart(self):
        """
        Resets the variables for the next round.
        """

        self.n = 0
        self.Q_a = self.ini


class StationaryBandit(Bandit):

    def __init__(self, ini=0):

        super(StationaryBandit, self).__init__(ini=ini)

    def recalculate_estimated_reward(self, reward):
        """
        Recalculate the estimated reward according to the last yielded reward, as an average.
        """
        self.Q_a = self.Q_a + (1/self.n)*(reward - self.Q_a)

    def selected(self):
        """
        Steps executed if the bandit is selected. Update variables for the next iteration.
        """

        self.increased_selected_times()

        reward = self.yield_reward()
        if self.n == 1:
            self.Q_a = reward
        else:
            self.recalculate_estimated_reward(reward)

        return reward


class NonStationaryBandit(Bandit):

    def __init__(self, alpha, ini=0):

        super(NonStationaryBandit, self).__init__(ini=ini)

        self.alpha = alpha
        self.original_q_star = self.q_star

    def recalculate_estimated_reward(self, reward):
        """
        Recalculate the estimated reward according to the last yielded reward, acording to the step-size alpha
        """
        self.Q_a += self.alpha*(reward - self.Q_a)

    def rw_q_star(self):
        """
        Modify the real reward of the Bandit as a random walk with Normal distribution (mean 0, variance 0.0001)
        """
        self.q_star += np.random.normal(0, 0.01)

    def selected(self):
        """
        Steps executed if the bandit is selected. Update variables for the next iteration.
        """

        reward = self.yield_reward()

        self.increased_selected_times()
        self.recalculate_estimated_reward(reward)

        return reward

    def restart(self):
        """
        Resets the variables for the next round.
        """

        self.n = 0
        self.Q_a = self.ini
        self.q_star = self.original_q_star
