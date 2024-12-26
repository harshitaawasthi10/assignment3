import numpy as np

# Ridge Regression Function
def ridge_regression(X, y, lambda_reg):
    """
    X: Feature matrix (n_samples, n_features)
    y: Target vector (n_samples,)
    lambda_reg: Regularization strength
    """
    n_features = X.shape[1]
    I = np.eye(n_features)  # Identity matrix
    # Closed-form solution: w = (X^T X + lambda * I)^-1 X^T y
    w = np.linalg.inv(X.T @ X + lambda_reg * I) @ X.T @ y
    return w

# User Inputs
n_features = int(input("Enter the number of features: "))
n_data_points = int(input("Enter the number of data points: "))

print("Enter the feature values row-wise (one row for each data point):")
X = []
for _ in range(n_data_points):
    row = list(map(float, input().split()))
    X.append(row)
X = np.array(X)

y = []
print("Enter the target values (one per data point):")
for _ in range(n_data_points):
    y.append(float(input()))
y = np.array(y)

# Regularization parameter
lambda_reg = float(input("Enter the regularization strength (lambda): "))

# Perform Ridge Regression
weights = ridge_regression(X, y, lambda_reg)

# Output the best-fitting plane
print("\nThe coefficients (weights) of the best-fitting plane are:")
print(weights)
