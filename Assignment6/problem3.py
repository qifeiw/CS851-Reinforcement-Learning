import numpy as np
import matplotlib.pyplot as plt

# Define the range of s values
s_values = np.arange(0, 1.1, 0.1)

# Define the optimal value function v*(s)
def v_star(s):
    return np.minimum(np.minimum(2 * s, -0.1 * s + 0.2), 0.8 - s)

v_star_values = v_star(s_values)

# Feature matrix for 3 linear features
def linear_features(s_values):
    return np.vstack([np.ones_like(s_values), s_values, s_values**2]).T

# Feature matrix for 2 polynomial features with degree 2
def poly_features_degree2(s_values):
    return np.vstack([s_values, s_values**2]).T

# Feature matrix for 4 polynomial features with degree 3
def poly_features_degree3(s_values):
    return np.vstack([np.ones_like(s_values), s_values, s_values**2, s_values**3]).T

# Function to compute w*
def compute_w_star(Phi, v_star_values):
    return np.linalg.inv(Phi.T @ Phi) @ (Phi.T @ v_star_values)

# Function to plot phi(s)^T w
def plot_results(Phi, w_star, title):
    s_plot = np.linspace(0, 1, 100)
    phi_plot = Phi(s_plot)
    y_values = phi_plot @ w_star
    
    plt.plot(s_plot, y_values, label='Prediction')
    plt.plot(s_values, v_star_values, 'o', label='True v*(s)')
    plt.title(title)
    plt.xlabel('s')
    plt.ylabel('Phi(s)^T w')
    plt.legend()
    plt.show()

# Compute and plot for 3 linear features
Phi_linear = linear_features(s_values)
w_star_linear = compute_w_star(Phi_linear, v_star_values)
plot_results(linear_features, w_star_linear, "3 Linear Features")

# Compute and plot for 2 polynomial features (degree 2)
Phi_poly2 = poly_features_degree2(s_values)
w_star_poly2 = compute_w_star(Phi_poly2, v_star_values)
plot_results(poly_features_degree2, w_star_poly2, "2 Polynomial Features (Degree 2)")

# Compute and plot for 4 polynomial features (degree 3)
Phi_poly3 = poly_features_degree3(s_values)
w_star_poly3 = compute_w_star(Phi_poly3, v_star_values)
plot_results(poly_features_degree3, w_star_poly3, "4 Polynomial Features (Degree 3)")