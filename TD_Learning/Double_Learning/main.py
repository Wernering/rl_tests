# Reinforcement Learning An Introduction Page 134. Double Learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


class Agent:

    def __init__(self, update_method="Double Q-Learning", b_card=10):

        # Parameters
        self.epsilon = 0.1
        self.alpha = 0.1
        self.b_card = b_card

        # Update method Q-Learning or Double Q-Learning
        self.update = update_method

        # Nodes
        self.Terminal = {"Q1": np.zeros(1),
                         "Q2": np.zeros(1)}

        self.B = {"actions": {i: self.Terminal for i in range(b_card)},
                  "Q1": np.zeros(b_card),
                  "Q2": np.zeros(b_card)}

        self.A = {"actions": {0: self.Terminal,
                              1: self.B},
                  "Q1": np.zeros(2),
                  "Q2": np.zeros(2)}

        # Initial State. The States are 0:A, 1:B
        self.state = self.A

        # Initial pandas dataframe
        self.df = pd.DataFrame()

    def restart(self):
        self.A["Q1"] = np.zeros(2)
        self.A["Q2"] = np.zeros(2)

        self.B["Q1"] = np.zeros(self.b_card)
        self.B["Q2"] = np.zeros(self.b_card)

    def reward(self):
        """
        Any action taken in A gives back a reward of 0.
        Any action taken in B gives back a reward of N(-0.1, 1)
        """
        if self.state == self.A:
            return 0

        else:
            return np.random.normal(-0.1, 1)

    def choose_action(self, state):
        """
        Under the condition of the experiment, we choose the action according to the sum of expected values for Q1 and
        Q2.
        In case of Q-Learning, Q2 values will always be 0, so this function is equally valid.
        """

        if np.random.random() <= self.epsilon:
            arg = np.random.choice(list(state["actions"].keys()))
            return arg, state["actions"][arg]

        b = state["Q1"] + state["Q2"]
        argmax = np.random.choice(np.flatnonzero(b == b.max()))
        return argmax, state["actions"][argmax]

    def q_learning_update(self, initial_state, action, end_state):
        initial_state["Q1"][action] = initial_state["Q1"][action] + self.alpha * (self.reward() + max(end_state["Q1"]) -
                                                                                  initial_state["Q1"][action])

    def double_q_learning_update(self, initial_state, action, end_state):
        if np.random.random() < 0.5:
            argmax = np.random.choice(np.flatnonzero(end_state["Q2"] == end_state["Q2"].max()))
            initial_state["Q1"][action] = initial_state["Q1"][action] + self.alpha * (
                    self.reward() + end_state["Q1"][argmax] -
                    initial_state["Q1"][action])
        else:
            argmax = np.random.choice(np.flatnonzero(end_state["Q1"] == end_state["Q1"].max()))
            initial_state["Q2"][action] = initial_state["Q2"][action] + self.alpha * (
                    self.reward() + end_state["Q2"][argmax] -
                    initial_state["Q2"][action])

    def time_step(self):

        action, next_state = self.choose_action(self.state)

        if self.update == "Double Q-Learning":
            self.double_q_learning_update(self.state, action, next_state)

        else:
            self.q_learning_update(self.state, action, next_state)

        previous_state = self.state
        self.state = next_state

        return previous_state, action

    def episode(self):

        while self.state != self.Terminal:
            state, action = self.time_step()
        self.state = self.A

        return state, action

    def cycle(self, episodes=300):

        list_actions = []
        for ep in range(episodes):
            left_picked = 0

            state, action = self.episode()

            # If the previous-to-end state is B, means that Agent picked Left on A.
            if state == self.B:
                left_picked = 1

            list_actions.append(left_picked)

        return list_actions

    def run(self, runs=10000):

        for run in range(1, runs + 1):

            self.restart()

            left_picked_run = self.cycle()

            self.df[run] = left_picked_run

        self.modify_df(runs)

    def modify_df(self, runs):
        self.df["percentage_left_actions_from_A"] = (self.df == 1).sum(axis=1).div(runs)
        self.df = self.df["percentage_left_actions_from_A"]


if __name__ == "__main__":

    t1 = time.time()
    DQL = Agent(update_method="Double Q-Learning")
    DQL.run(runs=10000)
    t2 = time.time()
    print(f"Execution time Double-Q-Learning: {t2 - t1} seconds")

    QL = Agent(update_method="Q-Learning")
    QL.run(runs=10000)
    print(f"Execution time Q-Learning: {time.time() - t2} seconds")

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle("Double-Q-Learning vs Q-Learning")

    DQL.df.plot(ax=ax, legend=True)
    QL.df.plot(ax=ax, legend=True)

    ax.legend(["Double-Q-Learning", "Q-Learning"])

    plt.xlabel("Episodes")
    plt.ylabel("% left Actions from A")
    plt.show()
