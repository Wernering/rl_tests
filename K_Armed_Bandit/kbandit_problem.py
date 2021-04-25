from K_Armed_Bandit import StationaryBandit, NonStationaryBandit
import pandas as pd
import random
import copy


class KBanditProblem:

    def __init__(self, k, iteration, rounds, epsilon=0, initial_guess=0, stationary=True, alpha=0.5):

        self.k = k
        self.epsilon = epsilon
        self.iterations = iteration
        self.rounds = rounds
        self.initial_value = initial_guess
        self.stationary = stationary

        self.bandits = dict()

        if stationary:
            for i in range(k):
                self.bandits[i] = self.create_stationary_bandit(initial_guess)

        else:
            for i in range(k):
                self.bandits[i] = self.create_nonstationary_bandit(alpha, initial_guess)

        self.estimates = self.evaluate_all_bandits()
        self.results = pd.DataFrame()
        self.current_iteration = 1
        self.current_round = 1

    @staticmethod
    def create_stationary_bandit(ini=0):
        """
        Creates an object of a stationary bandit
        """
        return StationaryBandit(ini)

    @staticmethod
    def create_nonstationary_bandit(alpha, ini=0):
        """
        Creates an object of a non stationary bandit with alpha step-size
        """
        return NonStationaryBandit(alpha, ini)

    @staticmethod
    def random_selection(choices):
        """
        Returns a random value out of a list
        """
        return random.choice(choices)

    def evaluate_bandit(self, a):
        """
        Returns the value of the estimated reward for the bandit
        """
        return self.bandits[a].Q_a

    def evaluate_all_bandits(self):
        """
        Returns a diccionary with the key of the bandit and the estimated reward
        """
        return {x: self.evaluate_bandit(x) for x in self.bandits.keys()}

    def greedy_selection(self):
        """
        Returns the bandit with the highest estimated reward value (chooses randomly if there are 2 equals)
        """
        max_value = max(self.estimates.values())
        choices = [k for k, v in self.estimates.items() if v == max_value]

        if len(choices) == 1:
            return choices[0]

        else:
            return self.random_selection(choices)

    def non_greedy_selection(self):
        """
        Returns a random bandit. Not caring about estimated reward value
        """
        return self.random_selection(list(self.bandits.keys()))

    def non_greedy_action(self):
        """
        Checks a random generated number between [0, 1). if the number is inferior to epsilon, returns True
        """
        return random.random() <= self.epsilon

    def select_bandit(self):
        """
        If the random generated number is smaller or equal to epsilon, a non-greedy bandit is selected. otherwise a
        greedy selection is made
        """
        if self.non_greedy_action():
            return self.non_greedy_selection()

        else:
            return self.greedy_selection()

    def save_reward_iteration(self, reward):
        """
        Save the reward of the current iteration and current round in a dataframe
        """
        self.results.at[self.current_iteration, self.current_round] = reward

    def increase_iteration(self):
        """
        Adds 1 to the current iteration number
        """
        self.current_iteration += 1

    def increase_round(self):
        """
        Increase one to the current round number
        """
        self.current_round += 1

    def iteration(self):
        """
        Iteration of selection of bandit and update of it.
        """
        # Update estimated reward values
        self.estimates = self.evaluate_all_bandits()

        a = self.select_bandit()

        # Obtain real reward and update bandit
        reward = self.bandits[a].selected()

        return reward

    def reset_bandits(self):
        """
        Reset the bandits to the initial state.
        """
        for a in self.bandits:
            self.bandits[a].restart()

    def round(self):
        """
        Cycle of the object in one round (i iterations). Get the real reward (and recalculate the estimated reward value
        of the bandit), save it ad increase the number of iteration.
        """

        self.current_iteration = 0
        for i in range(1, 1 + self.iterations):
            reward = self.iteration()
            self.save_reward_iteration(reward)
            self.increase_iteration()

            if not self.stationary:
                self.recalculate_nonstationary_q_star()

    def recalculate_nonstationary_q_star(self):
        """
        In case of a nonstationary bandit problem, all the bandits recalculate their real q_star value after each
        iteration
        """
        for k, b in self.bandits.items():
            b.rw_q_star()

    def experiment(self):
        """
        Cycle that executes a whole experiment (n rounds of i iterations). After every round, the bandits are reseted,
        that means, the Q_a = initial value and number of selections = 0, but the real reward value is kept
        """
        for i in range(1, 1 + self.rounds):
            self.reset_bandits()
            self.round()
            self.current_round += 1

        self.average_results()

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

    def average_results(self):
        """
        Add a column of 'average' of reward value over the iterations
        """
        self.results['average'] = self.results.mean(axis=1)
