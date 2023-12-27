def update_q_value(exp_value: float, reward: int, exp_value_s: float, alpha: float, gamma: float) -> float:
    """
    exp_value: Q(S,A): Expected value given state S and action A
    reward: R: Reward optained when taken action A in state S
    alpha: step-size parameter
    gamma: discounted rate parameter
    exp_value_s: Q(S',a): Expected value given state S', taking action a. The way to obtain this value dependes on the update strategy

        Q-Learning: max_a (Q(S',a))
        Double Q-Learning: Q(S', argmax_a(Q'(S', a)))
        SARSA: Q(S', a'), where a' is obtained following policy π
        Expected SARSA: E_π(Q(S',a' | S'))

    """
    return exp_value + alpha * (reward + gamma * exp_value_s - exp_value)
