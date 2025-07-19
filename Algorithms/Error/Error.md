# 📉 Error Types and Empirical Risk Minimization (ERM)

In machine learning, **understanding error** is crucial to evaluating how well a model performs — both on the data it has seen and, more importantly, on new unseen data.

This document explains the **main types of error** and the principle of **Empirical Risk Minimization (ERM)**, which is foundational to most supervised learning algorithms.

---

## 🧠 What Is "Error" in Machine Learning?

"Error" refers to how far off a model's predictions are from the actual values. We use **loss functions** (like MSE, log-loss, 0-1 loss) to compute it.

# Examples of function losses
| Nombre         | Fórmula                                     | Típico para...        | 
| -------------- | ------------------------------------------- | --------------------- |
| **MSE**        | $\frac{1}{n} \sum (y - \hat{y})^2$          | Regresión             |
| **MAE**        | $(\frac{1}{n} \sum y - \hat{y})$            |                       |
| **Log-loss**   | $-y \log(\hat{y}) - (1-y)\log(1 - \hat{y})$ | Clasificación binaria |
| **Hinge loss** | $\max(0, 1 - y \cdot \hat{y})$              | SVM                   |

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
$$
---

## 🔢 Key Types of Error

### 1. 🧠 **True Error** (Expected Risk / Generalization Error)

\[
\mathcal{R}(h) = \mathbb{E}_{(x, y) \sim \mathcal{D}} [\ell(h(x), y)]
\]

- Measures how well hypothesis \( h \) performs on the **entire data distribution** \( \mathcal{D} \).
- **Theoretical**: we can't compute it directly because we don't know the true distribution.
- Goal: **minimize this** to create a generalizable model.

---

### 2. 📊 **Empirical Error** (Empirical Risk)

\[
\hat{\mathcal{R}}(h) = \frac{1}{n} \sum_{i=1}^{n} \ell(h(x_i), y_i)
\]

- Average loss over the **training set**.
- **Practical approximation** of the true error.
- Used in most training algorithms to optimize model parameters.

---

### 3. 🔁 **Training Error**

- The empirical error **measured on the training data**.
- Usually low if the model is overfitting.

---

### 4. 🔬 **Validation Error**

- Error measured on a **validation set** (a held-out subset of data).
- Used for **tuning hyperparameters** and preventing overfitting.

---

### 5. 📦 **Test Error**

- Error measured on **unseen test data**.
- Provides a **realistic estimate** of how the model will perform in production.
- Should only be used **after** all tuning is done.

---

## 📐 Empirical Risk Minimization (ERM)

ERM is the core principle behind most learning algorithms:

> Choose the hypothesis \( h \in \mathcal{H} \) that **minimizes the empirical risk**.

Formally:

\[
h^* = \arg\min_{h \in \mathcal{H}} \hat{\mathcal{R}}(h)
\]

Where:
- \( \mathcal{H} \): hypothesis space (set of all models you're considering)
- \( \hat{\mathcal{R}}(h) \): empirical error on training data

---

### ⚠️ Problem: Overfitting in ERM

Minimizing empirical error **too perfectly** may lead the model to fit noise in the training data, resulting in **poor generalization** (high true error).

---

### ✅ Solution: Regularized ERM

\[
\hat{\mathcal{R}}_{reg}(h) = \hat{\mathcal{R}}(h) + \lambda \cdot \Omega(h)
\]

- \( \Omega(h) \): regularization term (e.g., \( \|\theta\|^2 \))
- \( \lambda \): regularization strength
- Encourages **simpler models** with better generalization

---

## 🧪 Practical Summary

| Type of Error     | Where Measured      | Purpose                                |
|-------------------|----------------------|-----------------------------------------|
| True Error         | Unknown data distribution | Theoretical goal (what we want to minimize) |
| Empirical Error    | Training set         | What we actually minimize (via ERM)     |
| Training Error     | Training set         | Indicates fit quality; low = possibly overfitting |
| Validation Error   | Validation set       | Used to tune model/hyperparameters      |
| Test Error         | Final test set       | Estimates true error on unseen data     |

---

## 🧩 When Is Each Error Used?

| Error Type        | When is it used?                                                                 |
|-------------------|----------------------------------------------------------------------------------|
| **True Error**     | ⚠️ *Theoretical only*. Used in theory to define generalization. Not computable. |
| **Empirical Error**| During training to guide optimization (via ERM). Used internally in loss minimization. |
| **Training Error** | To check how well the model fits the training data. Used to **detect overfitting** when compared to validation/test error. |
| **Validation Error** | During model selection (e.g., grid search, cross-validation). Used to choose the best **hyperparameters**. |
| **Test Error**     | After all tuning is done. Used for **final model evaluation** to estimate real-world performance. Should only be used **once**, not to guide training. |

---

## 📌 Final Notes

- ERM is at the heart of training ML models.
- Regularization and validation help bridge the gap between **empirical** and **true** error.
- Minimizing training error is **not enough** — we want **low test error**, which implies good generalization.
