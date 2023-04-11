def q_learning(
    exp_value: float, reward: int, exp_value_s: list, alpha: float, gamma: float
) -> float:
    """
    exp_value: Q(S,A): Expected value given state S and action A
    reward: R: Reward optained when taken action A in state S
    alpha: step-size parameter
    gamma: discounted rate parameter
    exp_value_s: Q(S',a): Expected value given state S', taking action a


    """
    return exp_value + alpha * (reward + gamma * max(exp_value_s) - exp_value)
