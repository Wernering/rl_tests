# Reinforcement Learning. An Introduction. Page 81
# Jack's Car Rental

import time
import math
import numpy as np
from operator import itemgetter
import seaborn as sns
import matplotlib.pyplot as plt
import os


class Location:

    def __init__(self, **kw):

        # maximum number of cars in location
        self.cardinality = kw.get("cap", 20)

        # Expected value of lease probability
        self.lease_k = kw.get("lk", 3)

        # Expected value of returned car probability
        self.return_k = kw.get("rk", 3)

        # Dictionary f probabilities of renting x cars
        self.prob_lease = self.dictionary_of_probabilities(self.lease_k)
        self.prob_return = self.dictionary_of_probabilities(self.return_k)

    @staticmethod
    def poisson(k, n):
        return math.exp(-k) * (k ** n) / math.factorial(n)

    def dictionary_of_probabilities(self, k):
        """
        Calculates a dictionary of probabilities of renting x cars. The dictionary is as big as the probability is over
        epsilon 0.001.
        """

        poisson_dictionary = dict()

        epsilon = 0.001
        x = 0

        while True:
            prob = self.poisson(k, x)

            if x <= k:
                poisson_dictionary[x] = prob

            else:
                if prob >= epsilon:
                    poisson_dictionary[x] = prob
                else:
                    break
            x += 1

        # Normalize probability
        extra = (1 - sum(poisson_dictionary.values())) / len(poisson_dictionary)
        poisson_dictionary = {k: v + extra for k, v in poisson_dictionary.items()}

        return poisson_dictionary


class JacksRental:

    def __init__(self):
        self.folder = "./lineal_problem"
        self.create_folder()

        # Maximum number of cars moved between locations
        self.max_move = 5

        # Small value that stops the Policy Iteration
        self.theta = 50

        # Cost of moving a car
        self.cost = -2

        # Reward of renting a car
        self.reward = 10

        # Discounting rate
        self.discount = 0.9

        # Location A
        self.A = Location(cap=20, lk=3, rk=3)
        self.B = Location(cap=20, lk=4, rk=2)

        # Policy Matrix
        self.policy_matrix = np.zeros([self.A.cardinality + 1, self.B.cardinality + 1]).astype(int)

        # Value Function Matrix
        self.VF_matrix = np.zeros([self.A.cardinality + 1, self.B.cardinality + 1])

    def create_folder(self):
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

    def expected_return(self, state, action, vf_matrix):
        """ for a given state, to evaluate the expected return, we assume that we make an action, and after taking the
        action, we lease an recover cars, according to the possibilities, ending in a state 2, result of the after
        environment resolution.

        :param state: State that is being evaluated
        :param action: number between -5 to 5. Indicates the number of cars from A to B.
        :param vf_matrix: matrix of value functions for every state

        :return: value function of state s.
        """
        xa, xb = state

        new_xa, new_xb = max(min(xa - action, self.A.cardinality), 0), max(min(xb + action, self.B.cardinality), 0)

        vf = 0

        # The cost of the action is 'discounted' from the reward
        vf += abs(action) * self.cost

        # We go through every number of posibilities that are higher than 0.1%.
        for l1, pl1 in self.A.prob_lease.items():
            for l2, pl2 in self.B.prob_lease.items():
                for r1, pr1 in self.A.prob_return.items():
                    for r2, pr2 in self.B.prob_return.items():
                        # Probability of those 4 events happening
                        probability = pl1 * pl2 * pr1 * pr2

                        # Number of valid leases (rental). Returns may be leased the day after
                        valid_xa, valid_xb = min(new_xa, l1), min(new_xb, l2)

                        # expected reward of getting to s'. Rt = r
                        r = (valid_xa + valid_xb) * self.reward

                        # State s'
                        aux_xa, aux_xb = [max(min(new_xa - valid_xa + r1, self.A.cardinality), 0),
                                          max(min(new_xb - valid_xb + r2, self.B.cardinality), 0)]

                        # Bellmans Equation
                        vf += probability * (r + self.discount * vf_matrix[aux_xa][aux_xb])

        return vf

    def synchronous_policy_evaluation(self, vf_matrix, policy_matrix):
        """
        Evaluate the current policy in a synchronous way. That means, evaluate all the new value with the old
        value functions results

        :return: the updated matrix of values functions
        """
        max_diff = math.inf
        while max_diff >= self.theta:

            delta = 0
            old_vf = vf_matrix.copy()

            for state_a in range(vf_matrix.shape[0]):
                for state_b in range(vf_matrix.shape[0]):
                    vf_matrix[state_a][state_b] = self.expected_return([state_a, state_b],
                                                                       policy_matrix[state_a][state_b],
                                                                       old_vf)

            max_diff = max(delta, np.amax(abs(vf_matrix - old_vf)))

            print(max_diff)
        return vf_matrix

    def policy_improvement(self, vf_matrix, policy_matrix):
        """
        Function to change the current policy for the greedy option, given a value function for each state

        :param vf_matrix: matrix of current value function values
        :param policy_matrix: current policy matrix. The value will be replace for the new policy for each state

        :return: the new matrix of policies
        """

        policy_matrix = policy_matrix.copy()
        for state_a in range(vf_matrix.shape[0]):
            for state_b in range(vf_matrix.shape[0]):

                # Maximum number of cars that can be moved from A to B
                xab = min(state_a, self.max_move)

                # Maximum number of cars that can be moved from B to A
                xba = -min(state_b, self.max_move)

                vf_list = list()
                for action in range(xba, xab + 1):
                    vf_list.append((action, self.expected_return([state_a, state_b], action, vf_matrix)))

                # Update Policy
                policy_matrix[state_a][state_b] = max(vf_list, key=itemgetter(1))[0]

        return policy_matrix

    @staticmethod
    def save_matrix(matrix, folder, name):
        np.save("{}/{}".format(folder, name), matrix)

    @staticmethod
    def plot_solution(matrix, folder, name, annot=True):
        f, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(matrix, annot=annot, linewidths=.5, ax=ax)
        ax.invert_yaxis()
        ax.set(xlabel="Location 1", ylabel="Location 2")
        f.savefig("{}/{}".format(folder, name))

    def synchronous_solve(self):
        """
        Method to execute a synchronous solve of the algorithm
        """
        solved = False
        iteration = 0
        self.folder += "/sync_solution"
        self.create_folder()

        self.plot_solution(self.policy_matrix, self.folder, "policy_matrix_0.png")
        self.plot_solution(self.VF_matrix, self.folder, "vf_matrix_0.png", annot=False)

        while not solved:

            iter_init_time = time.time()

            # Evaluate Policy synchronously
            self.VF_matrix = self.synchronous_policy_evaluation(self.VF_matrix.copy(),
                                                                self.policy_matrix.copy())

            # Get new policy
            new_policy_matrix = self.policy_improvement(self.VF_matrix.copy(),
                                                        self.policy_matrix.copy())

            if np.array_equal(new_policy_matrix, self.policy_matrix):
                solved = True

            # update Policy
            self.policy_matrix = new_policy_matrix

            iteration += 1

            self.save_matrix(self.VF_matrix, self.folder, "vf{}".format(iteration))
            self.save_matrix(self.policy_matrix, self.folder, "policy{}".format(iteration))

            print("iteration: {}".format(iteration))
            print("Minimum value: {}\nMaximum value: {}".format(np.amin(self.VF_matrix), np.amax(self.VF_matrix)))

            # update theta
            self.theta /= 10

            # Save plots of iteration results
            self.plot_solution(self.policy_matrix, self.folder, "policy_matrix_{}.png".format(iteration))
            self.plot_solution(self.VF_matrix, self.folder, "vf_matrix_{}.png".format(iteration), annot=False)

            # Write down time of iteration calculation
            with open("{}/sol_info.txt".format(self.folder), "a+") as f:
                f.write("Iteration: {}, calculation time: {}\n".format(iteration, time.time() - iter_init_time))

    def solve(self, pe="synchronous"):

        init_time = time.time()

        if pe == "synchronous":
            self.synchronous_solve()

        # Write down total time of calculation
        with open("{}/sol_info.txt".format(self.folder), "a+") as f:
            f.write("Total time of resolution: {}\n\n".format(time.time() - init_time))


xx = JacksRental()
print(sum(xx.A.prob_lease.values()), sum(xx.B.prob_lease.values()), sum(xx.B.prob_return.values()))
xx.solve()
