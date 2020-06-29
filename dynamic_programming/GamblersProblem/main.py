# Reinforcement Learning. An Introduction. Page 84
# Gambler's Problem

import numpy as np
import math
import matplotlib.pyplot as plt
import time
import random
import os


class GamblerProblem:

    def __init__(self, ph):

        self.VF_matrix = np.zeros([101])

        self.policy_matrix = np.zeros([101]).astype(int)

        self.rewards = np.zeros([101])
        self.rewards[100] = 1

        self.theta = 0.01

        self.ph = ph
        self.name = str(self.ph).replace(".", "")

        self.folder = self.create_folder()

    @staticmethod
    def plot_solution(matrix, folder, name, bar=True):
        f, ax = plt.subplots(figsize=(9, 6))
        if bar:
            ax.bar(range(1, 100), matrix)
        else:
            ax.plot(range(1, 100), matrix)
        ax.set(xlabel="Location 1", ylabel="Location 2")
        f.savefig("{}/{}".format(folder, name))
        plt.close()

    def create_folder(self):
        if not os.path.isdir("./{}".format(self.name)):
            os.mkdir("./{}".format(self.name))

        if not os.path.isdir("./{}/policy".format(self.name)):
            os.mkdir("./{}/policy".format(self.name))

        if not os.path.isdir("./{}/value_function".format(self.name)):
            os.mkdir("./{}/value_function".format(self.name))

        return "./{}".format(self.name)

    def expected_return(self, state, action, vf_matrix):
        """
        :param state:
        :param action:
        :param vf_matrix:
        :return:
        """

        return self.ph * (self.rewards[state + action] + vf_matrix[state + action]) + (1 - self.ph) * (self.rewards[state - action] + vf_matrix[state - action])

    def one_step_lookahead(self, vf_matrix, state):
        """
        :param vf_matrix:
        :param state:

        :return:
        """

        # List f new possible values for each action given the current vf_matrix
        eval_vf = np.zeros(101)

        # for the state, evaluate all the possible actions
        for action in range(1, min(state, 100 - state) + 1):
            eval_vf[action] = self.expected_return(state, action, vf_matrix)

        # We return the max value calculated
        return eval_vf

    def solve(self):

        init_time = time.time()
        solved = False

        iteration = 0

        while not solved:

            iteration += 1
            init_time_iteration = time.time()

            # Dummy array
            new_vf = np.zeros(101)
            # Update all the values of the states with a one step lookahead
            for state in range(1, 100):
                new_vf[state] = np.max(self.one_step_lookahead(self.VF_matrix, state))

            delta = np.max(abs(self.VF_matrix - new_vf))

            self.VF_matrix = new_vf

            if delta < self.theta:
                solved = True

            self.plot_solution(self.VF_matrix[1:100], "./{}/value_function".format(self.folder),
                               "VF_{}.png".format(iteration), bar=False)

            with open("info_{}.txt".format(self.name), "a+") as f:
                f.write("Iteration: {}, calculation time: {}\n".format(iteration, time.time() - init_time_iteration))

        # Once there is no change in the value function matrix:
        for state in range(1, 100):
            actions = self.one_step_lookahead(self.VF_matrix, state)
            self.policy_matrix[state] = np.argmax(actions)

        self.plot_solution(self.policy_matrix[1:100], "./{}/policy".format(self.folder),
                           "optima_pol.png".format(iteration))

        print("Iteration: {}, policy: {}".format(iteration, self.policy_matrix))

        with open("info_{}.txt".format(self.name), "a+") as f:
            f.write("Total calculation time: {}\n\n".format(time.time() - init_time))


xx = GamblerProblem(ph=0.40)
xx.solve()
