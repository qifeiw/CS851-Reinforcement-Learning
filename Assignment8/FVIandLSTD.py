

import numpy as np


###########Step 1, Define the Domain, Transition function and Policy#############
# Parameters
gamma = 0.99  # Discount factor
states = np.arange(0, 101)  # States S = [0, 100]
actions = np.arange(0, 8)  # Actions A = {0, 1, ..., 7}

# Transition function
def f(s, a, w):
    return min(100, max(0, s + 11 - w - 13 * a))

# Stationary policy pi
def policy_pi(s):
    return round(7 * (s / 100))

#########Step 2, FVI to Approximate Value Function 
def fixed_value_iteration(policy, gamma=0.99, tol=1e-6):
    V = np.zeros(len(states))  # Initialize value function
    while True:
        delta = 0
        for s in states:
            a = policy(s)
            expected_value = 0
            for w in range(11):  # Disturbance w ~ U(0, 10)
                s_prime = f(s, a, w)
                reward = -(s - 50) ** 2 - abs(a - 3)
                expected_value += (1 / 11) * (reward + gamma * V[s_prime])
            delta = max(delta, abs(expected_value - V[s]))
            V[s] = expected_value
        if delta < tol:
            break
    return V

##########Step 3, Compute Close-to-Optimal Value Function Using FVI
def optimal_value_iteration(gamma=0.99, tol=1e-6):
    V = np.zeros(len(states))
    policy = np.zeros(len(states), dtype=int)
    while True:
        delta = 0
        for s in states:
            action_values = []
            for a in actions:
                expected_value = 0
                for w in range(11):
                    s_prime = f(s, a, w)
                    reward = -(s - 50) ** 2 - abs(a - 3)
                    expected_value += (1 / 11) * (reward + gamma * V[s_prime])
                action_values.append(expected_value)
            best_action_value = max(action_values)
            best_action = actions[np.argmax(action_values)]
            delta = max(delta, abs(best_action_value - V[s]))
            V[s] = best_action_value
            policy[s] = best_action
        if delta < tol:
            break
    return V, policy

########Step 4, Construction the Optimal Policy
def construct_policy_from_value_function(V, gamma=0.99):
    policy = np.zeros(len(states), dtype=int)
    for s in states:
        action_values = []
        for a in actions:
            expected_value = 0
            for w in range(11):
                s_prime = f(s, a, w)
                reward = -(s - 50) ** 2 - abs(a - 3)
                expected_value += (1 / 11) * (reward + gamma * V[s_prime])
            action_values.append(expected_value)
        best_action = actions[np.argmax(action_values)]
        policy[s] = best_action
    return policy

# Example usage
V_optimal, _ = optimal_value_iteration()
optimal_policy = construct_policy_from_value_function(V_optimal)
print("Optimal Policy:", optimal_policy)

########Step 5, LSTD for value function approximation
def feature_vector(s):
    return np.array([1, s, s ** 2])

def lstd_policy_evaluation(policy, gamma=0.99, tol=1e-6):
    n_features = 3  # Example with 3 polynomial features
    A = np.zeros((n_features, n_features))
    b = np.zeros(n_features)

    for s in states:
        a = policy(s)
        phi_s = feature_vector(s)
        expected_phi_next = np.zeros(n_features)
        expected_reward = 0

        for w in range(11):
            s_prime = f(s, a, w)
            phi_s_prime = feature_vector(s_prime)
            reward = -(s - 50) ** 2 - abs(a - 3)
            expected_phi_next += (1 / 11) * phi_s_prime
            expected_reward += (1 / 11) * reward

        A += np.outer(phi_s, phi_s - gamma * expected_phi_next)
        b += phi_s * expected_reward

    # Solve for the weight vector w: A * w = b
    w = np.linalg.solve(A, b)
    
    # Define value function approximation
    def approximate_value_function(s):
        return np.dot(w, feature_vector(s))

    return approximate_value_function

# Usage example
approx_value_func = lstd_policy_evaluation(policy_pi)

# Test the approximated value for a sample state
state_sample = 50
print(f"Approximated Value at state {state_sample}: {approx_value_func(state_sample)}")