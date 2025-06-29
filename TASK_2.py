import numpy as np

# Step 1: Get matrix size from user
n = int(input("Enter the number of rows and columns (square matrix): "))

# Step 2: Input matrix entries row by row
print(f"\nEnter the entries of the {n}x{n} matrix row by row, separated by space:")
A = []

for i in range(n):
    row = list(map(float, input(f"Row {i + 1}: ").strip().split()))
    while len(row) != n:
        print("Please enter exactly", n, "numbers.")
        row = list(map(float, input(f"Row {i + 1}: ").strip().split()))
    A.append(row)

A = np.array(A)

# Step 3: Diagonalization
eigenvalues, eigenvectors = np.linalg.eig(A)
D = np.diag(eigenvalues)
P = eigenvectors
P_inv = np.linalg.inv(P)
A_reconstructed = P @ D @ P_inv

# Step 4: Print results
print("\nOriginal Matrix A:\n", A)
print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors (P):\n", P)
print("\nDiagonal Matrix D:\n", D)
print("\nInverse of P (P^-1):\n", P_inv)
print("\nReconstructed A (P * D * P^-1):\n", A_reconstructed)
