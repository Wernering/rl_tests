# External
import numpy as np


class GamblersProblem:
    def __init__(self, states: int, ph: float) -> None:
        self.vf = np.zeros([states + 1])
        self.policy = np.zeros([states + 1]).astype(int)

        self.rewards = np.zeros([states + 1])
        self.rewards[states] = 1

        self.states = states
        self.ph = ph

    def calculate_expected_return(self, vf: np.ndarray, state: int, action: int) -> float:
        er = self.ph * (self.rewards[state + action] + vf[state + action])
        er += (1 - self.ph) * (self.rewards[state - action] + vf[state - action])
        return er

    def get_states(self) -> int:
        return self.states

    def update_vf(self, state: int, new_value: float) -> None:
        self.vf[state] = new_value

    def update_policy(self, state: int, new_action: int) -> None:
        self.policy[state] = new_action
