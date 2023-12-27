# Standard Library
import enum

# External
import numpy as np

# Project
from app.utils.equations import update_q_value

# Local
from .env import Node


class UpdateMethod(enum.Enum):
    QL = "Q-Learning"
    DQL = "Double Q-Learning"


class ActionValue:
    def __init__(self) -> None:
        self.Q1 = 0
        self.Q2 = 0

    def reset(self) -> None:
        self.Q1 = 0
        self.Q2 = 0

    def sum_q(self) -> float:
        return self.Q1 + self.Q2


class Agent:
    def __init__(self, nodes: list[Node], update_method=UpdateMethod, **kw):
        # Parameters
        self.epsilon = kw.pop("epsilon", 0.1)
        self.alpha = kw.pop("alpha", 0.1)
        self.gamma = kw.pop("gamma", 1)

        assert isinstance(update_method, UpdateMethod)
        self.update_method = update_method

        self.qvalues = {n.name: {a: ActionValue() for a in n.actions()} for n in nodes}

    def restart(self) -> None:
        for aux in self.qvalues.values():
            for action in aux.values():
                action.reset()

    def choose_action(self, state: Node) -> int:
        """
        Under the condition of the experiment, we choose the action according to the sum of expected values for Q1 and
        Q2.
        In case of Q-Learning, Q2 values will always be 0, so this function is equally valid.
        """

        if np.random.random() <= self.epsilon:
            action = np.random.choice(list(self.qvalues.get(state.name).keys()))
            return action

        max_v = max([a.sum_q() for a in self.qvalues.get(state.name).values()])
        action = np.random.choice([a for a, v in self.qvalues.get(state.name).items() if v.sum_q() == max_v])
        return action

    def update_q_values(self, initial_state: Node, action: int, end_state: Node, reward: float) -> None:
        if self.update_method == UpdateMethod.QL:
            exp_value = self.qvalues[initial_state.name][action].Q1
            exp_value_s = max([a.Q1 for a in self.qvalues.get(end_state.name).values()])  # <- Q-Learning

            self.qvalues[initial_state.name][action].Q1 = update_q_value(
                exp_value=exp_value, reward=reward, exp_value_s=exp_value_s, alpha=self.alpha, gamma=self.gamma
            )

        elif self.update_method == UpdateMethod.DQL:
            if np.random.random() < 0.5:
                max_v = max([a.Q2 for a in self.qvalues.get(end_state.name).values()])
                argmax_Q2 = np.random.choice([a for a, v in self.qvalues.get(end_state.name).items() if v.Q2 == max_v])
                exp_value = self.qvalues[initial_state.name][action].Q1
                exp_value_s = self.qvalues[end_state.name][argmax_Q2].Q1  # <- Double Q-Learning

                self.qvalues[initial_state.name][action].Q1 = update_q_value(
                    exp_value=exp_value, reward=reward, exp_value_s=exp_value_s, alpha=self.alpha, gamma=self.gamma
                )

            else:
                max_v = max([a.Q1 for a in self.qvalues.get(end_state.name).values()])
                argmax_Q1 = np.random.choice([a for a, v in self.qvalues.get(end_state.name).items() if v.Q1 == max_v])
                exp_value = self.qvalues[initial_state.name][action].Q2
                exp_value_s = self.qvalues[end_state.name][argmax_Q1].Q2  # <- Double Q-Learning

                self.qvalues[initial_state.name][action].Q2 = update_q_value(
                    exp_value=exp_value, reward=reward, exp_value_s=exp_value_s, alpha=self.alpha, gamma=self.gamma
                )
