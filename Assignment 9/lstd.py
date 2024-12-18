import numpy as np

# Define the MDP
states = [0, 1]
actions = [0, 1]
transition_probs = {
    (0, 0): (1, 1),  # State 0, Action 0 -> State 1 with reward 1
    (0, 1): (0, 0),  # State 0, Action 1 -> State 0 with reward 0
    (1, 0): (0, 0),  # State 1, Action 0 -> State 0 with reward 0
    (1, 1): (1, 1)   # State 1, Action 1 -> State 1 with reward 1
}
policy = {0: [0.8, 0.2], 1: [0.8, 0.2]}  # High probability of Action 0

# Parameters for LSTD
gamma = 0.9
alpha = 0.01
num_episodes = 100

# Function to generate dataset
def generate_dataset(policy, num_episodes, on_policy=True):
    dataset = []
    for _ in range(num_episodes):
        state = np.random.choice(states)
        for _ in range(10):  # Fixed-length episodes
            if on_policy:
                action = np.random.choice(actions, p=policy[state])
            else:
                action = np.random.choice(actions)  # Off-policy (random actions)
            next_state, reward = transition_probs[(state, action)]
            dataset.append((state, action, reward, next_state))
            state = next_state
    return dataset

# Generate datasets
D1 = generate_dataset(policy, num_episodes, on_policy=False)  # Off-policy
D2 = generate_dataset(policy, num_episodes, on_policy=True)   # On-policy

# LSTD function
def lstd(dataset):
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for (s, a, r, s_next) in dataset:
        phi_s = np.array([1 if s == 0 else 0, 1 if s == 1 else 0])
        phi_s_next = np.array([1 if s_next == 0 else 0, 1 if s_next == 1 else 0])
        A += np.outer(phi_s, phi_s - gamma * phi_s_next)
        b += phi_s * r
    try:
        theta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        theta = np.zeros(2)  # Divergence case
    return theta

# Apply LSTD to both datasets
theta_D1 = lstd(D1)
theta_D2 = lstd(D2)

print("LSTD result with off-policy dataset D1:", theta_D1)
print("LSTD result with on-policy dataset D2:", theta_D2)