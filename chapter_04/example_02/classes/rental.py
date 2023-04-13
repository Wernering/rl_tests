import logging
import numpy as np

from config.logger import LOG_NAME
from classes.location import Location

logger = logging.getLogger(LOG_NAME)


class JacksRental:
    """
    The state is the amount of cars at the end of the day in each Location.\n
    The action is the net number of cars moved from location A to B.
    """

    def __init__(
        self,
        max_moves: int,
        cost: float,
        reward: float,
        location_a: dict,
        location_b: dict,
        discount: float = 0.9,
        **kw,
    ):
        self.max_move = max_moves
        self.cost = -abs(cost)
        self.reward = reward
        self.discount = discount

        self.location_a = location_a
        self.location_b = location_b

        self.A = Location(name="A", **location_a)
        self.B = Location(name="B", **location_b)

        # Small value that stops the Policy Iteration
        self.theta = kw.get("theta", 50)

        # Policy Matrix
        self.policy_matrix = np.zeros([self.A.capacity + 1, self.B.capacity + 1])

        # Value Function Matrix
        self.state_value = self.policy_matrix.copy()
        self.policy_matrix = self.policy_matrix.astype(int)

    def get_max_state_value(self) -> float:
        return float(np.amax(self.state_value))

    def get_min_state_value(self) -> float:
        return float(np.amin(self.state_value))

    def update_theta(self, division: int | float) -> None:
        self.theta /= division

    def calculate_expected_value(self, state: tuple, action: int, v_matrix: np.ndarray):
        # initial state S
        init_car_a, init_car_b = state

        # initial state S' after the action
        init_car_a = max(min(init_car_a - action, self.A.capacity), 0)
        init_car_b = max(min(init_car_b + action, self.B.capacity), 0)

        # expected return
        v = 0
        v += abs(action) * self.cost

        # We must iterate over each possible escenario for a day (lease and return)
        # to calculate the expected return
        for lease_a, prob_lease_a in self.A.lease_probability.items():
            for lease_b, prob_lease_b in self.B.lease_probability.items():
                for return_a, prob_return_a in self.A.return_probability.items():
                    for return_b, prob_return_b in self.B.return_probability.items():
                        probability = (
                            prob_lease_a * prob_lease_b * prob_return_a * prob_return_b
                        )

                        leased_cars_a = min(lease_a, init_car_a)
                        leased_cars_b = min(lease_b, init_car_b)
                        reward = (leased_cars_a + leased_cars_b) * self.reward

                        # ending state S'
                        end_car_a = max(
                            min(init_car_a - leased_cars_a + return_a, self.A.capacity),
                            0,
                        )
                        end_car_b = max(
                            min(init_car_b - leased_cars_b + return_b, self.B.capacity),
                            0,
                        )

                        v += probability * (
                            reward + self.discount * v_matrix[end_car_a][end_car_b]
                        )
        return v

    def policy_evaluation(self) -> None:
        delta = np.inf
        while delta >= self.theta:
            delta = 0
            old_v = self.state_value.copy()
            for state_a in range(self.A.capacity + 1):
                for state_b in range(self.B.capacity + 1):
                    self.state_value[state_a][state_b] = self.calculate_expected_value(
                        state=(state_a, state_b),
                        action=self.policy_matrix[state_a][state_b],
                        v_matrix=old_v,
                    )
            delta = max(delta, float(np.amax(abs(self.state_value - old_v))))
            logger.debug(f"Theta Value: {self.theta}. Max difference: {delta}")

    def policy_improvement(self) -> bool:
        stable = True
        for state_a in range(self.A.capacity + 1):
            for state_b in range(self.B.capacity + 1):
                old_action = self.policy_matrix[state_a][state_b]

                # Maximum number of cars that can be moved from A to B
                max_ab = min(state_a, self.max_move)

                # Maximum number of cars that can be moved from B to A
                max_ba = -min(state_b, self.max_move)

                v_dict = {}
                for action in range(max_ba, max_ab + 1):
                    v_dict[action] = self.calculate_expected_value(
                        state=(state_a, state_b),
                        action=action,
                        v_matrix=self.state_value,
                    )
                # In case of more than one action as optimal, we take the first
                new_action = [
                    key
                    for key, value in v_dict.items()
                    if value == max(v_dict.values())
                ][0]

                # Update Policy
                self.policy_matrix[state_a][state_b] = new_action

                if new_action != old_action:
                    stable = False
        return stable
