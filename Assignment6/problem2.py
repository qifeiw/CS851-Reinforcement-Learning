import numpy as np

# Define the matrix A (3x2) and vector b (3x1)
A = np.array([[2, 3], 
              [3, 5], 
              [1, 1]])

b = np.array([1, 2, 3])

# Compute the least squares solution using the formula x = (A^T A)^(-1) A^T b
AtA_inv = np.linalg.inv(A.T @ A)  # (A^T A)^(-1)
Atb = A.T @ b                     # A^T b
x = AtA_inv @ Atb                 # x = (A^T A)^(-1) A^T b

# Print the solution
print(x)