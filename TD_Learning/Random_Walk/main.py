# Reinforcement Learning An Introduction page 125

import pandas as pd
import matplotlib.pyplot as plt
from random import random
import math
import time


class Node:
    def __init__(self, name):
        self.name = name
        self.V = 0.5

    def td_update(self, r, vs, a, y):
        """
        :param r: Inmediate Reward of action taken
        :param vs: Expected Reward of state S'
        :param a: Alpha value
        :param y: Gamma Value (Discount Rate)
        """
        self.V = self.V + a*(r + y*vs - self.V)

    def mc_update(self, g, a):
        """
        :param g: Gt. Final reward of the episode
        :param a: step size. Learning rate
        """
        self.V = self.V + a*(g - self.V)


class Model:
    def __init__(self, nodes, alpha, initial_node="default"):
        self.number_nodes = nodes
        self.a = alpha
        self.y = 1

        self.model_list = []

        # Create Initial Model
        self.initial_model(nodes)

        if initial_node == "default":
            self.pos = int(nodes/2)
        else:
            self.pos = int(initial_node)

    @staticmethod
    def create_node(name):
        return Node(name)

    @staticmethod
    def movement():
        if random() < 0.5:
            return -1
        else:
            return 1

    def initial_model(self, cant):
        for i, _ in enumerate(range(cant)):
            self.model_list.append(self.create_node(i))

    def td_update_node(self, node, r, vs):
        node.td_update(r, vs, self.a, self.y)

    def td_cycle(self):

        while True:
            action = self.movement()

            # If we are on the further left node and we move to the left (-1)
            if self.pos == 0 and action == -1:
                r = 0
                vs = 0
                self.td_update_node(self.model_list[self.pos], r, vs)
                return

            # If we are on the further right node and we move to the right (+1)
            elif self.pos == len(self.model_list) - 1 and action == 1:
                r = 1
                vs = 0
                self.td_update_node(self.model_list[self.pos], r, vs)
                return

            # Any other cases
            else:
                vs = self.model_list[self.pos + action].V
                r = 0
                self.td_update_node(self.model_list[self.pos], r, vs)
                self.pos += action

    def td_repeat(self, cycles):

        results = pd.DataFrame(columns=range(self.number_nodes))

        for cycle in range(cycles):

            # Execute episode
            self.td_cycle()

            results.loc[cycle] = [x.V for x in self.model_list]

        return results

    def mc_update_node(self, node, g):
        node.mc_update(g, self.a)

    def mc_cycle(self):

        nodes = set()
        while True:
            # Add current node to the set of nodes of the episode
            nodes.add(self.model_list[self.pos])

            # Take an action
            action = self.movement()

            # If we are on the further left node and we move to the left (-1)
            if self.pos == 0 and action == -1:
                r = 0
                for node in nodes:
                    self.mc_update_node(node, r)
                return

            # If we are on the further right node and we move to the right (+1)
            elif self.pos == len(self.model_list) - 1 and action == 1:
                r = 1
                for node in nodes:
                    self.mc_update_node(node, r)
                return

            # Any other cases
            else:
                self.pos += action

    def mc_repeat(self, cycles):

        results = pd.DataFrame(columns=range(self.number_nodes))

        for cycle in range(cycles):
            # Execute episode
            self.mc_cycle()

            results.loc[cycle] = [x.V for x in self.model_list]

        return results


if __name__ == "__main__":

    init_time = time.time()
    real = [1/6, 2/6, 3/6, 4/6, 5/6]

    for alpha in [0.15, 0.1, 0.05]:

        errors_list = []
        for run in range(100):
            xx = Model(5, alpha=alpha)
            steps = xx.td_repeat(100)

            errors = steps.sub(real, axis='columns')
            errors = errors.applymap(lambda x: x**2)
            errors = errors.sum(axis='columns')
            errors = errors.div(5)
            errors = errors.apply(lambda x: math.sqrt(x))

            errors_list.append(errors)

        errors = pd.concat(errors_list, axis=1)
        errors = errors.mean(axis="columns")
        errors.plot(label=f"""TD alpha {alpha}""")

    for alpha in [0.01, 0.02, 0.04]:
        errors_list = []
        for run in range(100):
            xx = Model(5, alpha=alpha)
            steps = xx.mc_repeat(100)

            errors = steps.sub(real, axis='columns')
            errors = errors.applymap(lambda x: x**2)
            errors = errors.sum(axis='columns')
            errors = errors.div(5)
            errors = errors.apply(lambda x: math.sqrt(x))

            errors_list.append(errors)

        errors = pd.concat(errors_list, axis=1)
        errors = errors.mean(axis="columns")
        errors.plot(label=f"""MC alpha {alpha}""")

    plt.legend()

    print(f"Tiempo de Ejecucion: {time.time() - init_time} segundos")
    plt.show()

