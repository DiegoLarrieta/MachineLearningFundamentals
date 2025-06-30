"""
✅ Objetivo
Entrenar un modelo de regresión lineal utilizando gradient descent para encontrar los valores óptimos de β₀ y β₁.


"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Dataset (Years of experience vs Salary in thousands)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([35, 40, 50, 55, 60, 65, 70, 85, 90, 95])
n = len(X)

# 2. Initialize parameters
beta_0 = 0  # Intercept
beta_1 = 0  # Slope

# 3. Hyperparameters
alpha = 0.01   # Learning rate
epochs = 1000  # Number of iterations

# 4. Store loss for visualization
loss_history = []

# 5. Gradient Descent Loop
for epoch in range(epochs):
    y_pred = beta_0 + beta_1 * X
    error = y_pred - y

    # Gradients
    grad_b0 = (1/n) * np.sum(error)
    grad_b1 = (1/n) * np.sum(error * X)

    # Parameter update
    beta_0 -= alpha * grad_b0
    beta_1 -= alpha * grad_b1

    # Compute loss (MSE)
    mse = (1/n) * np.sum(error**2)
    loss_history.append(mse)

    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: β₀ = {beta_0:.2f}, β₁ = {beta_1:.2f}, MSE = {mse:.2f}")

# 6. Final model
print(f"\nFinal model: y = {beta_0:.2f} + {beta_1:.2f}x")

# 7. Plot loss curve
plt.plot(range(epochs), loss_history)
plt.title("Loss (MSE) over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.show()

# 8. Plot regression result
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, beta_0 + beta_1 * X, color='red', label='Predicted line (GD)')
plt.title("Linear Regression using Gradient Descent")
plt.xlabel("Years of Experience")
plt.ylabel("Salary (k USD)")
plt.legend()
plt.grid(True)
plt.show()
