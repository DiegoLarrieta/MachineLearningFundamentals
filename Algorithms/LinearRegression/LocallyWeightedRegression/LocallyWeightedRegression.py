import numpy as np
import matplotlib.pyplot as plt

# 1. Generate a nonlinear dataset
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.1, 100)  # y = sin(x) + noise

# Convert X to 2D shape for matrix operations
X = X.reshape(-1, 1)

# 2. Add a column of 1s to X for the intercept (bias term)
def add_bias(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))

# 3. Gaussian kernel function
def get_weights(x_query, X, tau):
    m = X.shape[0]
    weights = np.exp(-np.sum((X - x_query)**2, axis=1) / (2 * tau**2))
    return np.diag(weights)

# 4. Locally Weighted Regression function
def locally_weighted_regression(X, y, tau, x_query):
    X_bias = add_bias(X)
    xq_bias = np.array([[1, x_query]])

    W = get_weights(np.array([[x_query]]), X, tau)
    
    # θ = (Xᵀ W X)⁻¹ Xᵀ W y
    theta = np.linalg.pinv(X_bias.T @ W @ X_bias) @ X_bias.T @ W @ y
    y_pred = xq_bias @ theta
    return y_pred[0]

# 5. Predict for many points
tau = 0.5  # Bandwidth (you can change this)
X_test = np.linspace(0, 10, 200)
y_pred = np.array([locally_weighted_regression(X, y, tau, xq) for xq in X_test])

# 6. Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Training Data", alpha=0.5)
plt.plot(X_test, y_pred, color="red", label=f"LWR Prediction (τ={tau})", linewidth=2)
plt.title("Locally Weighted Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
