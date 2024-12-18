import pandas as pd
import numpy as np

Q1 = {}
Q2 = {}

# Load data
samples = pd.read_csv("samples_example.csv")
for _, row in samples.iterrows():
    state = (row["Step"], row["Balance"], row["Positive"], row["Negative"])
    action = row["Action"]
    reward = row["Reward"]
    next_state = (row["Step"] + 1, row["Balance"], row["Positive"], row["Negative"])

    if np.random.rand() < 0.5:
        max_next_q = Q2.get((next_state, np.argmax([Q1.get((next_state, a), 0) for a in actions])), 0)
        Q1[(state, action)] = Q1.get((state, action), 0) + alpha * (reward + gamma * max_next_q - Q1.get((state, action), 0))
    else:
        max_next_q = Q1.get((next_state, np.argmax([Q2.get((next_state, a), 0) for a in actions])), 0)
        Q2[(state, action)] = Q2.get((state, action), 0) + alpha * (reward + gamma * max_next_q - Q2.get((state, action), 0))
