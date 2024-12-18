import pandas as pd
import numpy as np

# Load data
samples = pd.read_csv("samples_example.csv")

# Initialize parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
actions = samples["Action"].unique()
Q_table = {}  # Tabular Q-learning

# Q-learning with tabular approach
for _, row in samples.iterrows():
    state = (row["Step"], row["Balance"], row["Positive"], row["Negative"])
    action = row["Action"]
    reward = row["Reward"]
    next_state = (row["Step"] + 1, row["Balance"], row["Positive"], row["Negative"])

    max_next_q = max(Q_table.get((next_state, a), 0) for a in actions)
    Q_table[(state, action)] = Q_table.get((state, action), 0) + alpha * (reward + gamma * max_next_q - Q_table.get((state, action), 0))