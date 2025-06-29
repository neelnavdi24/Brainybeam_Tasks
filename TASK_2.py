import numpy as np

# Step 1: Define a square matrix A
A = np.array([[4, 1],
              [2, 3]])

# Step 2: Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Step 3: Construct diagonal matrix D using eigenvalues
D = np.diag(eigenvalues)

# Step 4: P is the matrix of eigenvectors
P = eigenvectors

# Step 5: Compute the inverse of P
P_inv = np.linalg.inv(P)

# Step 6: Verify the diagonalization: A ≈ P * D * P_inv
A_reconstructed = P @ D @ P_inv

# Print results
print("Original Matrix A:\n", A)
print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors (P):\n", P)
print("\nDiagonal Matrix D:\n", D)
print("\nInverse of P (P^-1):\n", P_inv)
print("\nReconstructed A (P * D * P^-1):\n", A_reconstructed)
