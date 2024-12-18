import numpy as np

# Constants
gamma = 0.99
states = np.arange(101)  # S = {0, ..., 100}
actions = np.arange(8)   # A = {0, ..., 7}
w_disturbances = np.arange(11)  # w ∈ {0, ..., 10}

# Reward function
def reward(s, a):
    return -(s - 50) ** 2 - abs(a - 3)

# Transition function
def transition(s, a, w):
    next_state = round(min(100, max(0, s + 11 - w - 13 * a)))
    return next_state

# Policy π(s) = round(7 * s / 100)
def policy(s):
    return round(7 * s / 100)

# Value Iteration to find optimal value function V*
def value_iteration():
    V = np.zeros(101)  # Initialize value function
    policy = np.zeros(101, dtype=int)  # Initialize policy
    threshold = 1e-6  # Convergence threshold
    max_iterations = 1000
    
    for iteration in range(max_iterations):
        delta = 0
        for s in states:
            v = V[s]
            action_values = np.zeros(8)
            
            for a in actions:
                total_value = 0
                for w in w_disturbances:
                    s_next = transition(s, a, w)
                    total_value += (1 / 11) * (reward(s, a) + gamma * V[s_next])
                action_values[a] = total_value
            
            V[s] = max(action_values)
            policy[s] = np.argmax(action_values)
            delta = max(delta, abs(v - V[s]))
        
        if delta < threshold:
            break
            
    return V, policy

# Evaluate the stationary policy
def evaluate_policy(policy):
    V = np.zeros(101)
    max_iterations = 1000
    threshold = 1e-6
    
    for iteration in range(max_iterations):
        delta = 0
        for s in states:
            v = V[s]
            a = policy(s)
            total_value = 0
            for w in w_disturbances:
                s_next = transition(s, a, w)
                total_value += (1 / 11) * (reward(s, a) + gamma * V[s_next])
            V[s] = total_value
            delta = max(delta, abs(v - V[s]))
        
        if delta < threshold:
            break
            
    return V

# Main code execution
if __name__ == "__main__":
    # 1. Evaluate the stationary policy
    print("Evaluating the stationary policy...")
    V_pi = evaluate_policy(policy)
    print("Value function for the stationary policy:", V_pi)

    # 2. Compute the optimal value function using value iteration
    print("Computing the optimal value function using value iteration...")
    V_optimal, optimal_policy = value_iteration()
    print("Optimal value function:", V_optimal)
    print("Optimal policy:", optimal_policy)

    # 3. Verify that V_pi ≈ V_optimal
    assert np.allclose(V_pi, V_optimal, atol=1e-2), "Value functions do not match!"
    print("Numerical verification successful: V_pi ≈ V_optimal")