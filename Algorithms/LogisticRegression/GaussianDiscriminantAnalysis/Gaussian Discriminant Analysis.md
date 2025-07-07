# 🧠 Gaussian Discriminant Analysis (GDA)

**Gaussian Discriminant Analysis (GDA)** is a generative learning algorithm used for **classification**.  
It assumes that the data for each class is generated from a **Gaussian (normal) distribution**, and uses Bayes' theorem to classify new examples.

---

## 📚 What Is GDA?

GDA models the **probability distribution of each class** in the input space.  
It uses these distributions to compute the **posterior probability** of each class given a new input, and classifies based on the most probable one.

It is a **generative model**, unlike logistic regression, which is discriminative.

---

## 📐 Mathematical Assumptions

For binary classification (\( y \in \{0, 1\} \)), GDA assumes:

- \( x \mid y = 0 \sim \mathcal{N}(\mu_0, \Sigma) \)
- \( x \mid y = 1 \sim \mathcal{N}(\mu_1, \Sigma) \)

Where:
- \( \mu_0, \mu_1 \) = class-specific means
- \( \Sigma \) = shared covariance matrix for both classes
- \( \phi \) = prior probability of class 1: \( \phi = P(y = 1) \)

---

## 🔁 Training (Fitting the Parameters)

Given a dataset \( \{(x^{(i)}, y^{(i)})\}_{i=1}^n \):

1. Estimate class priors:
\[
\phi = \frac{1}{n} \sum_{i=1}^{n} y^{(i)}
\]

2. Estimate means:
\[
\mu_0 = \frac{1}{n_0} \sum_{i: y^{(i)} = 0} x^{(i)} \quad,\quad
\mu_1 = \frac{1}{n_1} \sum_{i: y^{(i)} = 1} x^{(i)}
\]

3. Estimate covariance matrix:
\[
\Sigma = \frac{1}{n} \sum_{i=1}^{n} (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T
\]

---

## 🔮 Prediction

To predict a new label for \( x \), compute:

\[
P(y = k \mid x) \propto P(x \mid y = k) P(y = k)
\]

And classify as:

\[
\hat{y} = \arg\max_{k \in \{0, 1\}} P(y = k \mid x)
\]

> Because the class-conditional distributions are Gaussians, these probabilities can be computed in closed form.

---

## ✅ Advantages

- ✔️ Works well when data actually follows Gaussian distributions
- ✔️ Has a **closed-form solution** for training — no iterative optimization required
- ✔️ Often **outperforms logistic regression** when assumptions hold

---

## ❌ Disadvantages

- ❌ Assumes that data is **normally distributed** for each class — can be too strong
- ❌ Shared covariance matrix might not reflect real-world separability
- ❌ Sensitive to outliers and poorly estimated parameters in small datasets

---

## 🧪 GDA vs Logistic Regression

| Feature                    | GDA (Generative)              | Logistic Regression (Discriminative) |
|----------------------------|-------------------------------|--------------------------------------|
| Models \( P(x \mid y) \)? | ✅ Yes                        | ❌ No                                |
| Models \( P(y \mid x) \)? | ✅ (via Bayes rule)           | ✅ Directly                          |
| Training                  | Closed-form (analytic)         | Gradient-based (numerical)          |
| Output                    | Posterior probabilities        | Posterior probabilities              |
| Assumptions               | Gaussian distribution          | Linear boundary                      |

---

## 📊 Use Cases

- Binary or multiclass classification
- Spam detection
- Medical diagnosis
- Email classification
- When features are believed to be **Gaussian-like** per class

---

## 🧠 Summary

| Property          | Value                          |
|-------------------|--------------------------------|
| Type              | Supervised, Generative         |
| Input             | Feature vectors + class labels |
| Output            | Class predictions              |
| Core Assumption   | Normal (Gaussian) distributions|
| Training          | Closed-form parameter estimates|

---

## 💡 Final Note

> GDA is elegant, efficient, and powerful **when its assumptions match the data**.  
> It’s a great algorithm to understand both probability theory and machine learning foundations.

