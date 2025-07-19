import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Crear datos simulados
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
true_func = lambda x: 2 * x + 3
y = true_func(X) + np.random.normal(0, 2, size=X.shape)

# 2. Dividir en training, validation y test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

# 3. Entrenar modelo
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predicciones
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
y_full_pred = model.predict(X)  # Para estimar el true error

# 5. Calcular errores
errors = {
    "Empirical (Training)": mean_squared_error(y_train, y_train_pred),
    "Validation": mean_squared_error(y_val, y_val_pred),
    "Test": mean_squared_error(y_test, y_test_pred),
    "True (Estimated)": mean_squared_error(y, y_full_pred),
    "ERM Objective": mean_squared_error(y_train, y_train_pred),  # mismo que el entrenamiento
}

# 6. Imprimir errores
print("📊 Error Summary:")
for name, value in errors.items():
    print(f"{name}: {value:.4f}")

# 7. Visualizar errores
plt.figure(figsize=(10, 6))
bars = plt.bar(errors.keys(), errors.values(), color=["#4CAF50", "#2196F3", "#FFC107", "#9C27B0", "#607D8B"])
plt.title("Comparison of Error Types in ML")
plt.ylabel("Mean Squared Error (MSE)")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Agregar etiquetas encima de las barras
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()
