
import pandas as pd
import numpy as np

def softmax_policy(state, theta):
    logits = [np.dot(theta, phi(state, a)) for a in actions]
    exp_logits = np.exp(logits - np.max(logits))  # Stability trick
    return exp_logits / exp_logits.sum()

def update_theta(samples, theta, alpha):
    for _, row in samples.iterrows():
        state = (row["Step"], row["Balance"], row["Positive"], row["Negative"])
        action = row["Action"]
        reward = row["Reward"]

        grad_log_pi = phi(state, action) - np.dot(softmax_policy(state, theta), [phi(state, a) for a in actions])
        theta += alpha * grad_log_pi * reward
