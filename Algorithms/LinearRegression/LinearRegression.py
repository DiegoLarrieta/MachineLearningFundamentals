"""
🎯 Objetivo del ejemplo:
Predecir el salario de una persona en función de sus años de experiencia laboral usando regresión lineal simple.

Es un caso realista, fácil de visualizar y muy usado en entrevistas técnicas para introducir modelos supervisados.

📦 Dataset:
Vamos a crearlo nosotros. Usaremos un pequeño dataset simulado con una tendencia clara, ideal para entender visualmente el modelo.

Qué representa:

β₁ (pendiente): cuánto aumenta el salario por cada año adicional de experiencia

β₀ (intercepto): salario estimado cuando la experiencia es cero
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Dataset (same as before)
data = {
    'YearsExperience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [35, 40, 50, 55, 60, 65, 70, 85, 90, 95]
}

df = pd.DataFrame(data)

# 2. Visualize raw data
plt.scatter(df['YearsExperience'], df['Salary'], color='blue')
plt.title('Salary vs Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary (k USD)')
plt.grid(True)
plt.show()

# 3. Split data into features and labels
X = df['YearsExperience'].values
y = df['Salary'].values

# 4. Train/test split (20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Manually calculate β₁ and β₀ (closed-form solution)
x_mean = np.mean(X_train)
y_mean = np.mean(y_train)

numerator = np.sum((X_train - x_mean) * (y_train - y_mean))
denominator = np.sum((X_train - x_mean) ** 2)

beta_1 = numerator / denominator
beta_0 = y_mean - beta_1 * x_mean

print(f"Intercept (β₀): {beta_0:.2f}")
print(f"Coefficient (β₁): {beta_1:.2f}")

# 6. Predict on test set
y_pred = beta_0 + beta_1 * X_test

# 7. Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 8. Visualize regression line
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, beta_0 + beta_1 * X, color='red', label='Regression Line (manual)')
plt.title('Linear Regression from Scratch')
plt.xlabel('Years of Experience')
plt.ylabel('Salary (k USD)')
plt.legend()
plt.grid(True)
plt.show()
