import numpy as np
import matplotlib.pyplot as plt

def gambler_value_iteration_with_T(ph, gama=0.9, T=20, theta=1e-9):
    states = np.arange(1, 101)

    # Initiallize value functions, 0 for all states except V(100)
    V = np.zeros(101)
    V[100] = 1

    # Policy, initialized to stake 0 for all states
    policy = np.zeros(101)

    # Value iteration for T iterations
    for t in range(T):
        for s in range(1, 100):
            old_v = V[s]
            action_values = []

            for a in range(1, min(s, 100-s) + 1):
                win_state = s + a
                lose_state = s - a
                action_value = ph * V[win_state] + (1-ph) * V[lose_state]
                action_values.append(action_value)

            V[s] = max(action_values)
            policy[s] = np.argmax(action_values) + 1
            delta = max(delta, abs(old_v - V[s]))

            if delta < theta:
                break
    return V, policy

# Plotting the results
def plot_gambler(V, policy, ph, T):
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot value function
    ax[0].plot(V, label=f'p_h = {ph}, T = {T}')
    ax[0].set_xlabel('Capital')
    ax[0].set_ylabel('Value Estimates')
    ax[0].set_title(f'Value Function after {T} Iterations')
    ax[0].grid(True)
    
    # Plot policy
    ax[1].plot(policy, label=f'p_h = {ph}, T = {T}')
    ax[1].set_xlabel('Capital')
    ax[1].set_ylabel('Final Policy (stake)')
    ax[1].set_title(f'Final Policy after {T} Iterations')
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# Solve for ph = 0.25 and T = 20
ph = 0.25
T = 20
V, policy, iterations = gambler_value_iteration_with_T(ph, T=T)
plot_gambler(V, policy, ph, T)

# Solve for ph = 0.55 and T = 20
ph = 0.55
V, policy, iterations = gambler_value_iteration_with_T(ph, T=T)
plot_gambler(V, policy, ph, T)