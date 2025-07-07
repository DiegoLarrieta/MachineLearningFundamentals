import numpy as np
import matplotlib.pyplot as plt

# 1. Toy dataset (2D)
# Two features: [x1, x2] → binary label (0 or 1)
X = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [2.0, 2.5],
    [3.0, 3.5],
    [3.5, 3.0],
    [4.0, 4.5],
    [5.0, 6.0],
    [5.5, 5.8],
    [6.0, 6.2],
    [6.5, 5.5],
])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 5 from class 0, 5 from class 1

# 2. Estimate phi (prior)
phi = np.mean(y)  # P(y=1)
print(f"phi (P(y=1)): {phi:.2f}")

# 3. Estimate class means
mu0 = X[y == 0].mean(axis=0)
mu1 = X[y == 1].mean(axis=0)
print("mu0 (mean class 0):", mu0)
print("mu1 (mean class 1):", mu1)

# 4. Estimate shared covariance matrix
n = X.shape[0]
Sigma = np.zeros((2, 2))
for i in range(n):
    x_i = X[i].reshape(-1, 1)
    mu_i = mu1.reshape(-1, 1) if y[i] == 1 else mu0.reshape(-1, 1)
    Sigma += (x_i - mu_i) @ (x_i - mu_i).T
Sigma /= n
print("Shared covariance matrix (Σ):\n", Sigma)

# 5. Define Gaussian density function
def gaussian_density(x, mu, sigma):
    size = x.shape[0]
    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)
    norm_const = 1.0 / (np.power((2 * np.pi), size / 2) * np.sqrt(det))
    x_mu = x - mu
    result = norm_const * np.exp(-0.5 * (x_mu.T @ inv @ x_mu))
    return result

# 6. Predict function using Bayes Rule
def predict(x):
    x = x.reshape(-1, 1)
    p0 = gaussian_density(x, mu0.reshape(-1, 1), Sigma) * (1 - phi)
    p1 = gaussian_density(x, mu1.reshape(-1, 1), Sigma) * phi
    return 1 if p1 > p0 else 0

# 7. Test the model
print("\nPredictions:")
test_points = np.array([
    [2.5, 2.8],
    [3.0, 3.0],
    [5.2, 5.9],
    [6.2, 6.0]
])
for point in test_points:
    label = predict(point)
    print(f"{point} → Class {label}")

# 8. Visualize decision boundary
def plot_decision_boundary(X, y, mu0, mu1):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1')
    plt.scatter(mu0[0], mu0[1], color='black', marker='x', s=100, label='mu0')
    plt.scatter(mu1[0], mu1[1], color='black', marker='o', s=100, label='mu1')

    # Grid of points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = np.array([predict(p) for p in grid]).reshape(xx.shape)

    # Contour plot
    plt.contourf(xx, yy, preds, alpha=0.2, levels=[-1, 0, 1], colors=['blue', 'red'])
    plt.title("Gaussian Discriminant Analysis (GDA)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_decision_boundary(X, y, mu0, mu1)
