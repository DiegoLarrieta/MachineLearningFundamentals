"""
🧠 ¿Por qué usamos Gradient Ascent en Logistic Regression?
📍 Porque Logistic Regression no minimiza el error cuadrático, sino que maximiza una función llamada log-likelihood.

Así que:

Usar Gradient Ascent sobre log-likelihood ✅

O usar Gradient Descent sobre negative log-likelihood ✅
Ambos caminos son equivalentes.


"""

import numpy as np

# 1. Sigmoid function (activation)
# Transforms linear outputs into probabilities in the range (0, 1)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 2. Prediction function using the logistic regression model
# Computes the linear combination of inputs and weights, then applies sigmoid
def predict(X, weights):
    z = np.dot(X, weights)  # Linear combination (z = Xw)
    return sigmoid(z)       # Convert to probability

# 3. Log-likelihood function (our objective to maximize)
# Measures how well the current weights explain the actual labels
def compute_log_likelihood(X, y, weights):
    y_pred = predict(X, weights)
    # Add epsilon (1e-9) to prevent log(0)
    return np.sum(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))

# 4. Training function using Batch Gradient Ascent
# Iteratively adjusts weights to maximize log-likelihood
def gradient_ascent(X, y, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)  # Initialize weights to 0

    for epoch in range(epochs):
        y_pred = predict(X, weights)  # Predict probabilities with current weights
        gradient = np.dot(X.T, y - y_pred)  # Compute gradient of log-likelihood
        weights += learning_rate * gradient  # Update weights (gradient ascent step)

        # Print progress every 100 epochs
        if epoch % 100 == 0:
            ll = compute_log_likelihood(X, y, weights)
            print(f"Epoch {epoch}: Log-Likelihood = {ll:.4f}")

    return weights  # Final optimized weights

# 5. Toy dataset: binary classification (pass/fail based on study hours)
X_raw = np.array([
    [0.5],
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0],
    [7.0],
    [8.0],
    [9.0]
])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # Label: 1 = pass, 0 = fail

# 6. Add bias term (intercept) to the input features
X = np.hstack((np.ones((X_raw.shape[0], 1)), X_raw))  # Shape: (10, 2)

# 7. Train the logistic regression model
weights = gradient_ascent(X, y, learning_rate=0.1, epochs=1000)

# 8. Show final learned weights
print("\nFinal weights:", weights)

# 9. Classification function (uses a threshold of 0.5)
def classify(X, weights, threshold=0.5):
    return (predict(X, weights) >= threshold).astype(int)

# 10. Test the model on new inputs (hours studied)
test_hours = np.array([[1], [3], [5], [7], [9]])
test_X = np.hstack((np.ones((test_hours.shape[0], 1)), test_hours))  # Add bias
predictions = classify(test_X, weights)

# 11. Display predictions
print("\nTest predictions (hours studied):")
for h, p in zip(test_hours.flatten(), predictions):
    print(f"{h} hours → {'Pass' if p == 1 else 'Fail'}")
