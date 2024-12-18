import numpy as np

num_boxes = 100
prob_success = 0.3
prob_fail = 0.7

# cost values for fuel (cf) and walking (cw)

cf_values = [0.03, 0.4]
cw_values = [0.2, 0.5]
max_iters = 1000

def value_iteration(cf, cw, num_boxes=100, tol=1e-6):
    V = np.zeros(num_boxes + 1)
    policy = np.zeros(num_boxes, dtype=int)

    for _ in range(max_iters):
        V_prev = V.copy()

        for s in range(1, num_boxes + 1):
            if s == num_boxes:
                park_value = prob_success * (-cw * s) + prob_fail * (-1 -cf)
                move_value = -1-cf
            else:
                park_value = prob_success * (-cw * s) + prob_fail * (V[s+1] -cf)
                move_value = V[s+1] -cf

            V[s] = min(park_value, move_value)
            policy[s-1] = 0 if park_value < move_value else 1

            if np.max(np.abs(V - V_prev)) < tol:
                break

        return V, policy
    
for cf, cw in zip(cf_values, cw_values):
    V_optimal, optimal_policy = value_iteration(cf, cw)
    print(f"For cf={cf}, cw={cw}:")
    print("Optimal Value Function:", V_optimal[1:])  # Skip terminal state value
    print("Optimal Policy (0=park, 1=move):", optimal_policy)
    print()