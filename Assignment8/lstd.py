import numpy as np
# Define constants
gamma = 0.9  # discount factor
# Define a small state space
S = [2, 4, 6]  
# Define the transition matrix under policy π (P^π)
# Rows are current states, columns are next states
P_pi = np.array([
    [0.5, 0.5, 0.0],
    [0.2, 0.5, 0.3],
    [0.0, 0.3, 0.7]
])
# Define the reward vector (r^π) for each state
r_pi = np.array([1.0, 0.5, 0.2])

Phi = np.array([
    [1, 0, 0],  # feature vector for state 2
    [0, 1, 0],  # feature vector for state 4
    [0, 0, 1]   # feature vector for state 6
])
# Compute Φ^T Φ
Phi_T_Phi = Phi.T @ Phi

# Compute Φ^T P^π Φ
Phi_T_P_Phi = np.zeros((Phi.shape[1], Phi.shape[1]))  # initialize with zeros

for s in range(len(S)):
    for s_prime in range(len(S)):
        Phi_T_P_Phi += P_pi[s, s_prime] * np.outer(Phi[s], Phi[s_prime])
# Compute Φ^T r^π
Phi_T_r_pi = Phi.T @ r_pi
# Solve for w using the LSTD formula
# w = (Φ^T Φ - γ * Φ^T P^π Φ)^-1 * Φ^T r^π
A = Phi_T_Phi - gamma * Phi_T_P_Phi
w = np.linalg.solve(A, Phi_T_r_pi)
# Display the result
print("Feature Weights (w):", w)