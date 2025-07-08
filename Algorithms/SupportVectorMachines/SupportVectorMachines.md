# 🧠 Support Vector Machines (SVM)

## 📌 What is an SVM?

A **Support Vector Machine** is a **supervised learning algorithm** used for **classification** and **regression** tasks — but it is most commonly used for **binary classification**.

Its goal is to **find the best decision boundary** (hyperplane) that separates data points of different classes **with the maximum margin**.

---

## ✏️ Core Idea

- Given a set of labeled training data \((x_i, y_i)\), where \(y_i \in \{-1, 1\}\), the SVM algorithm finds a **hyperplane** that separates the classes.

- A **hyperplane** in 2D is just a line. In 3D it's a plane. In higher dimensions, it’s called a hyperplane.

- SVM chooses the hyperplane that:
  - Separates the classes.
  - **Maximizes the margin**: the distance between the hyperplane and the **closest data points** from each class (called **support vectors**).

---

## 🧮 Mathematical Formulation

For **linearly separable** data:

We want to solve the optimization problem:

\[
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2
\]

subject to:

\[
y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1 \quad \forall i
\]

Where:
- \(\mathbf{w}\): weight vector (defines the orientation of the hyperplane)
- \(b\): bias term (defines the offset)
- The constraints ensure that the data is classified correctly with a margin of at least 1.

---

## 🔍 ¿Qué significa C en Support Vector Machines?
C es el parámetro de regularización en el SVM. Controla el balance entre:

Maximizar el margen (tener una separación clara entre clases)

Reducir errores de clasificación en los datos de entrenamiento

| Valor de `C`                    | ¿Qué hace?                                                | Comportamiento del modelo                       |
| --------------------------------| --------------------------------------------------------- | ----------------------------------------------- |
| **Pequeño** (`C ≈ 0.01`, `0.1`) | Permite más errores si con eso se obtiene un margen mayor | **Margen grande**, más **tolerancia a errores** |
| **Grande** (`C = 10`, `100`)    | Penaliza más los errores de clasificación                 | **Margen pequeño**, pero **menos errores**      |


C = small → soft margin SVM → más tolerancia a errores → menos overfitting
C = large → hard margin SVM → menos errores en training → más overfitting posible


## ❓ What if the data is not linearly separable?

### 1. Soft Margin SVM  
Allows **some misclassifications** to make the model more robust in practice.  
It introduces **slack variables** to permit margin violations while still penalizing them in the objective function.

### 2. Kernel Trick  
When data is not linearly separable, SVM can **map data to higher-dimensional space** using a **kernel function**, where a linear separator may exist.

#### Common kernel functions:

- **Linear Kernel**: \(\langle x, x' \rangle\)
- **Polynomial Kernel**: \((\langle x, x' \rangle + c)^d\)
- **RBF (Gaussian) Kernel**: \(\exp(-\gamma \|x - x'\|^2)\)
- **Sigmoid Kernel**: \(\tanh(\alpha \langle x, x' \rangle + c)\)

---

## ✅ Advantages

- Works well in **high-dimensional spaces**.
- Effective even when the number of features > number of samples.
- Robust to overfitting (especially with proper regularization).
- Powerful with **non-linear data** using kernels.

---

## ⚠️ Disadvantages

- Can be **computationally expensive** with large datasets.
- Requires careful tuning of hyperparameters (like `C`, kernel parameters).
- Less effective on **noisy datasets** with overlapping classes.

---

## 🧠 Use Cases

- **Text classification** (e.g., spam detection)
- **Image recognition**
- **Bioinformatics** (e.g., cancer classification)
- **Handwriting recognition**
- **Face detection**

---

## 🔍 Summary

| Feature                  | Description                                      |
|--------------------------|--------------------------------------------------|
| Type                    | Supervised (mainly classification)               |
| Strength                | Maximize margin, works in high-dimensional data  |
| Handles non-linearity   | Yes, via kernel trick                            |
| Common kernels          | Linear, RBF, Polynomial                          |
| Output                  | Hyperplane that separates classes                |
