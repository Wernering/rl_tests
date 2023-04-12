from classes.bandit import NonStationaryBandit
import random
import copy


class KBanditProblem:
    def __init__(
        self,
        k: int,
        epsilon: float = 0.0,
        initial_guess: float = 0.0,
        alpha=0.5,
    ):
        self.epsilon = epsilon
        self.initial_value = initial_guess

        self.bandits = {}
        for i in range(k):
            self.bandits[i] = NonStationaryBandit(alpha, initial_guess)

        self.estimates = self.get_all_estimates()
        self.results = []

    def get_all_estimates(self) -> dict:
        return {x: self.bandits[x].get_Qa() for x in self.bandits.keys()}

    def select_bandit(self) -> NonStationaryBandit:
        if random.random() < self.epsilon:
            return random.choice(list(self.bandits.keys()))

        return random.choice(
            [k for k, v in self.estimates.items() if v == max(self.estimates.values())]
        )

    def save_reward_iteration(self, reward: float) -> None:
        self.results.append(reward)

    def iteration(self):
        """
        Iteration of selection of bandit and update of it.
        """
        bandit = self.select_bandit()
        real_reward = self.bandits[bandit].selected()
        self.estimates[bandit] = self.bandits[bandit].get_Qa()
        self.save_reward_iteration(real_reward)

        self.recalculate_nonstationary_q_star()

    def recalculate_nonstationary_q_star(self):
        """
        In case of a nonstationary bandit problem, all the bandits recalculate
        their real q_star value after each iteration
        """
        for bandit in self.bandits.values():
            bandit.rw_q_star()

    def reset_bandits(self):
        """
        Reset the bandits to the initial state.
        """
        for bandit in self.bandits.values():
            bandit.restart()
        self.results = []
        self.estimates = self.get_all_estimates()

    def copy(self):
        """
        Returns a copy of the instance of the class.
        """
        return copy.deepcopy(self)

    def change_epsilon(self, epsilon):
        """
        Change epsilon value for a new one.
        """
        self.epsilon = epsilon

    def get_epsilon(self):
        return self.epsilon
