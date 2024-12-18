import numpy as np
import matplotlib.pyplot as plt

n = 10
P = np.random.dirichlet(np.ones(n), size=n)

mu = np.random.dirichlet(np.ones(n))

print("Transition Matrix P: \n", P)
print("Initial Distribution  μ: ", mu)


# Problem 2, check if the state distribution sums to 1 for k = 1, ..., 100

def evolve_distribution(P, mu, k=100):
    d_k = mu.copy()
    for i in range(k):
        d_k = d_k @ P
        print(f"Sum of state distribution at step {i + 1}: {np.sum(d_k): .6f}")

evolve_distribution(P,mu)


def plot_distribution_evolution(P, mu, k=100):
    d_k = mu.copy()
    distributions =[d_k]

    for i in range(k):
        d_k = d_k @ P
        distributions.append(d_k)

    distributions = np.array(distributions)
    for i in range(distributions.shape[1]):
        plt.plot(distributions[:,i], label=f"State {i}")

    plt.title("Evolution of State Distribution")
    plt.xlabel('Step')
    plt.ylabel("Probability")
    plt.legend()
    plt.show()