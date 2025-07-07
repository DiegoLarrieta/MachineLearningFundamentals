# 🔼 Gradient Ascent – Maximizing Objective Functions in Machine Learning

**Gradient Ascent** is an optimization algorithm used to **maximize** a function by moving in the direction of its gradient.  
While Gradient Descent is used to **minimize** functions (like loss), Gradient Ascent is used when we want to **maximize** something — like probabilities, likelihoods, or rewards.

---

## 📚 What is Gradient Ascent?

Gradient Ascent is an iterative optimization algorithm that adjusts the model's parameters in the direction of the **steepest ascent** — the direction where the function increases the most.

> It’s used to **maximize** functions, unlike Gradient Descent, which is used to minimize them.

---

## 📐 Formula

\[
\theta := \theta + \alpha \cdot \nabla_\theta J(\theta)
\]

Where:
- \( \theta \) = model parameters
- \( \alpha \) = learning rate (step size)
- \( J(\theta) \) = the objective function to **maximize**
- \( \nabla_\theta J(\theta) \) = gradient of the objective function with respect to the parameters

---

## 🧠 What Does It Do?

- Takes a **step forward in the direction of the gradient**
- Gradually **increases the value of the function**
- Stops when it reaches a **maximum** (local or global)

---

## 🚀 What Problems Does It Solve?

- Maximization problems in machine learning, such as:
  - **Log-likelihood** in logistic regression or Naive Bayes
  - **Expected reward** in reinforcement learning
  - **Probability densities** in probabilistic models

---

## 🔄 Gradient Ascent vs. Gradient Descent

| Feature                    | Gradient Descent                                        | Gradient Ascent                |
|----------------------------|---------------------------------------------------------|--------------------------------|
| Goal                       | Minimize a function                                     | Maximize a function            |
| Update rule                | \( \theta := \theta - \alpha \cdot \nabla J(\theta) \)  | \( \theta := \theta + \alpha \cdot \nabla J(\theta) \) |
| Common usage               | Minimize loss/error                                     | Maximize likelihood, reward    |
| Visual metaphor            | Go downhill                                             | Go uphill                      |

> 🔁 **Both algorithms use the gradient**, but apply it in **opposite directions** depending on the objective.

---

## ✅ Advantages

- ✔️ Great for **probabilistic models** where maximizing a likelihood function is key
- ✔️ Conceptually simple and easy to implement
- ✔️ Works well when the objective function is differentiable

---

## ❌ Disadvantages

- ❌ Can converge to **local maxima**, not always the global one
- ❌ Requires careful **tuning of the learning rate**
- ❌ May not work well with **non-smooth or noisy functions**
- ❌ Sensitive to initialization

---

## 📦 Use Cases & Examples

### 1. **Logistic Regression (Maximum Likelihood Estimation)**

Maximizing the likelihood of data given parameters:

\[
\ell(\theta) = \sum_{i=1}^n \log P(y_i | x_i, \theta)
\]

Instead of minimizing a loss, we **maximize** the log-likelihood using Gradient Ascent.

### 2. **Naive Bayes**  
Involves maximizing probabilities (though in practice it's done analytically).

### 3. **Reinforcement Learning**  
Gradient Ascent is used in algorithms like **Policy Gradient** to maximize **expected cumulative reward**.

---

## 🧪 Example in Pseudocode

