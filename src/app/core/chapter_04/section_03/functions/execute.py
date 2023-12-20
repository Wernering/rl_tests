# Project
from app.utils import ctx_timer

# Local
from ..classes.rental import JacksRental
from ..config import LOGGER


def synchronous_solve(problem: JacksRental, theta_update: int | float) -> JacksRental:
    solved = False
    iteration = 0

    while not solved:
        iteration += 1

        LOGGER.info(f"Starting Iteration {iteration}")
        with ctx_timer(f"Iteration {iteration}", logger=LOGGER):
            # Evaluate Policy synchronously
            problem.policy_evaluation()

            # Get new policy
            solved = problem.policy_improvement()

        log_text = f"Iteration {iteration}: Max value: {problem.get_max_state_value()},"
        log_text += f" Min value: {problem.get_min_state_value()}"
        LOGGER.info(log_text)

        # update theta
        problem.update_theta(theta_update)

    return problem
