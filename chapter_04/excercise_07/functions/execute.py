import logging

from config.logger import LOG_NAME
from classes.rental import JacksRental
from functions.utils import timer

logger = logging.getLogger(LOG_NAME)


def synchronous_solve(problem: JacksRental, theta_update: int | float) -> JacksRental:
    solved = False
    iteration = 0

    while not solved:
        iteration += 1

        logger.info(f"Starting Iteration {iteration}")
        with timer(f"Iteration {iteration}"):
            # Evaluate Policy synchronously
            problem.policy_evaluation()

            # Get new policy
            solved = problem.policy_improvement()

        log_text = f"Iteration {iteration}: Max value: {problem.get_max_state_value()},"
        log_text += f" Min value: {problem.get_min_state_value()}"
        logger.info(log_text)

        # update theta
        problem.update_theta(theta_update)

    return problem
