import numpy as np
import matplotlib.pyplot as plt

def gambler_value_iteration_converge(ph, gamma=1, theta=1e-9):
    ''' Value iteration for Gambler's ruin problem until convergence.'''
    # States are capital from 0 to 100
    states = np.arange(101)
    
    # Initialize value function V(s) for all states
    V = np.zeros(101)  
    V[100] = 1  # Terminal state at s = 100, where V(100) = 1 (win)

    # Initialize policy
    policy = np.zeros(101)
    
    # Perform value iteration until convergence
    iterations = 0
    while True:
        delta = 0
        iterations += 1
        for s in range(1, 100):  # Exclude terminal states 0 and 100
            old_v = V[s]
            # Compute the Bellman update for the current state s
            action_values = []
            for a in range(1, min(s, 100 - s) + 1):  # Possible stakes
                win_state = s + a
                lose_state = s - a
                action_value = ph * V[win_state] + (1 - ph) * V[lose_state]
                action_values.append(action_value)
            
            # Update value function for state s
            V[s] = max(action_values)
            # Update policy for state s (best action to take)
            policy[s] = np.argmax(action_values) + 1  # +1 because action starts from 1
            
            # Track maximum change for convergence
            delta = max(delta, abs(old_v - V[s]))
        
        # Stop if the change in value function is below the threshold
        if delta < theta:
            break

    return V, policy, iterations  # Return value function, policy, and number of iterations

# Plotting function
def plot_gambler_converged(V, policy, ph):
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot value function
    ax[0].plot(V, label=f'p_h = {ph}')
    ax[0].set_xlabel('Capital')
    ax[0].set_ylabel('Value Estimates')
    ax[0].set_title(f'Optimal Value Function (Converged) for p_h = {ph}')
    ax[0].grid(True)
    
    # Plot policy
    ax[1].plot(policy, label=f'p_h = {ph}')
    ax[1].set_xlabel('Capital')
    ax[1].set_ylabel('Final Policy (stake)')
    ax[1].set_title(f'Optimal Policy (Converged) for p_h = {ph}')
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# Solve for ph = 0.25 with convergence (T -> infinity)
ph = 0.25
V_converged_025, policy_converged_025, iterations_025 = gambler_value_iteration_converge(ph)
print(f"Value iteration converged after {iterations_025} iterations for p_h = {ph}")
plot_gambler_converged(V_converged_025, policy_converged_025, ph)

# Solve for ph = 0.55 with convergence (T -> infinity)
ph = 0.55
V_converged_055, policy_converged_055, iterations_055 = gambler_value_iteration_converge(ph)
print(f"Value iteration converged after {iterations_055} iterations for p_h = {ph}")
plot_gambler_converged(V_converged_055, policy_converged_055, ph)