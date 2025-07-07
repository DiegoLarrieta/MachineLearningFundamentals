import numpy as np

# 1. Sigmoid function (activation)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 2. Predict probabilities using current weights
def predict(X, weights):
    z = np.dot(X, weights)
    return sigmoid(z)

# 3. Gradient (first derivative of log-likelihood)
def compute_gradient(X, y, weights):
    y_pred = predict(X, weights)
    return np.dot(X.T, y - y_pred)

# 4. Hessian matrix (second derivative of log-likelihood)
def compute_hessian(X, weights):
    y_pred = predict(X, weights)
    diag = y_pred * (1 - y_pred)  # diagonal entries of W matrix
    W = np.diag(diag)
    return -np.dot(X.T, np.dot(W, X))  # Hessian: -Xᵀ W X

# 5. Newton's method for logistic regression
def newtons_method(X, y, epochs=10, tolerance=1e-6):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)  # Initialize weights to 0

    for epoch in range(epochs):
        grad = compute_gradient(X, y, weights)
        hess = compute_hessian(X, weights)

        # Add small identity to avoid singular matrix
        hess_inv = np.linalg.pinv(hess)

        update = np.dot(hess_inv, grad)
        weights -= update  # Newton's step

        log_likelihood = np.sum(y * np.log(predict(X, weights) + 1e-9) + 
                                (1 - y) * np.log(1 - predict(X, weights) + 1e-9))
        
        print(f"Epoch {epoch}: Log-Likelihood = {log_likelihood:.4f}")
        
        if np.linalg.norm(update) < tolerance:
            print("Converged.")
            break

    return weights

# 6. Small dataset (study hours → pass/fail)
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
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# 7. Add bias term to X
X = np.hstack((np.ones((X_raw.shape[0], 1)), X_raw))  # shape (10, 2)

# 8. Train model using Newton's Method
weights = newtons_method(X, y)

print("\nFinal weights:", weights)

# 9. Classifier
def classify(X, weights, threshold=0.5):
    return (predict(X, weights) >= threshold).astype(int)

# 10. Test predictions
test_hours = np.array([[1], [3], [5], [7], [9]])
test_X = np.hstack((np.ones((test_hours.shape[0], 1)), test_hours))
predictions = classify(test_X, weights)

print("\nTest predictions (hours studied):")
for h, p in zip(test_hours.flatten(), predictions):
    print(f"{h} hours → {'Pass' if p == 1 else 'Fail'}")
