from random import random, choice
import numpy as np

from functions.algorithm import q_learning


class Agent:
    def __init__(self, rows: int, columns: int, movements: int, **kw) -> None:
        self._movements = movements

        self.alpha = kw.get("alpha", 0.1)
        self.gamma = kw.get("gamma", 0.1)
        self.epsilon = kw.get("epsilon", 0.1)

        self.steps = kw.get("steps", 10)

        self.rng = np.random.default_rng(kw.get("seed", None))

        self.expected_value = np.zeros([movements, rows, columns])
        self.model: dict[tuple[int, int], dict[int, tuple]] = {}

    def take_action(self, state_s: np.ndarray) -> int:
        state_s = tuple(state_s)
        if random() <= self.epsilon:
            return choice(range(0, self._movements))
        evaluations = self.expected_value[:, *state_s]
        best_movements = [i for i, v in enumerate(evaluations) if v == max(evaluations)]
        return choice(best_movements)

    def update_model(
        self, state_s: tuple, action: int, reward: int, state_s1: tuple
    ) -> None:
        if state_s not in self.model:
            self.model[state_s] = {}
        self.model[state_s][action] = reward, state_s1

    def model_learning(self) -> None:
        for _ in range(self.steps):
            state = tuple(self.rng.choice(list(self.model.keys())))
            action = self.rng.choice(list(self.model[state].keys()))
            reward, state_s = self.model[state][action]
            self.update_expected_value(
                state_s=state, action=action, reward=reward, state_s1=state_s
            )

    def update_expected_value(
        self, state_s: tuple, action: int, reward: int, state_s1: tuple
    ) -> None:
        exp_value = self.expected_value[action, *state_s]
        exp_value_s = self.expected_value[:, *state_s1]
        self.expected_value[action, *state_s] = q_learning(
            exp_value=exp_value,
            alpha=self.alpha,
            gamma=self.gamma,
            reward=reward,
            exp_value_s=exp_value_s,
        )
