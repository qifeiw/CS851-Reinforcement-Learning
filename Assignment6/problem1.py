import numpy as np

# Define the coefficient matrix A
A = np.array([[2, 3, 5],
              [3, 5, 2],
              [1, 1, 1]])

# Define the result vector b
b = np.array([1, 2, 3])

# Solve the system of equations using the inverse of A
x = np.linalg.inv(A).dot(b)
print(x)
