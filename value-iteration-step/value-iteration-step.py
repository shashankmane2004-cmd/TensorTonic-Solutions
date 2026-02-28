import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    values = np.array(values, dtype=float)
    transitions = np.array(transitions, dtype=float)
    rewards = np.array(rewards, dtype=float)

    S, A, _ = transitions.shape
    new_values = []

    for s in range(S):
        best = -float("inf")
        for a in range(A):
            expected_value = np.sum(transitions[s][a] * values)
            q = rewards[s][a] + gamma * expected_value
            if q > best:
                best = q
        new_values.append(best)

    return new_values