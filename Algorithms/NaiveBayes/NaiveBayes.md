# 📚 Naive Bayes Classifier

**Naive Bayes** is a family of probabilistic classifiers based on **Bayes’ Theorem**, with the strong assumption that the features are **conditionally independent** given the class.

It is widely used in machine learning for **text classification**, **spam detection**, **sentiment analysis**, and more due to its **simplicity**, **speed**, and surprisingly good performance.

---

## 🧠 Core Idea

Naive Bayes models the **posterior probability** of a class given input features using:

\[
P(y \mid x_1, x_2, ..., x_n) = \frac{P(x_1, ..., x_n \mid y) P(y)}{P(x_1, ..., x_n)}
\]

But assumes **conditional independence**, meaning:

\[
P(x_1, ..., x_n \mid y) = \prod_{i=1}^{n} P(x_i \mid y)
\]

So the formula simplifies to:

\[
P(y \mid x_1, ..., x_n) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)
\]

This is what makes it *naive* — it assumes that features are independent.

---

## 🔍 Types of Naive Bayes

| Type             | Description                                  | Use Case                        |
|------------------|----------------------------------------------|----------------------------------|
| **Gaussian**      | Features are continuous and Gaussian         | Iris dataset, sensor data       |
| **Multinomial**   | Features are counts (e.g. word frequencies)  | Text classification             |
| **Bernoulli**     | Features are binary (0/1)                    | Spam detection, binary features |

---

## ⚙️ Steps in Naive Bayes

1. **Estimate prior probability** for each class \( P(y) \)
2. **Estimate likelihood** for each feature given a class \( P(x_i \mid y) \)
3. Apply Bayes’ rule to get posterior probabilities
4. Predict the class with the **highest posterior**

---

## ✅ Advantages

- ✔️ Fast to train and predict
- ✔️ Works well with high-dimensional data
- ✔️ Robust to irrelevant features
- ✔️ Performs surprisingly well on real-world tasks

---

## ❌ Disadvantages

- ❌ Strong independence assumption rarely holds
- ❌ Struggles with continuous correlated features
- ❌ Can perform poorly when features are highly dependent

---

## 🎯 Applications

- Spam email detection
- Sentiment analysis (positive/negative)
- Document categorization
- Medical diagnosis
- Recommender systems (Naive Bayes collaborative filtering)

---

## 🧪 Example: Text Classification

Imagine you're classifying reviews as **positive** or **negative**:

- Estimate:
  - \( P(\text{positive}) \), \( P(\text{negative}) \)
  - \( P(\text{word} \mid \text{positive}) \), \( P(\text{word} \mid \text{negative}) \)

Then for a new review:
\[
P(\text{positive} \mid \text{review}) \propto P(\text{positive}) \cdot \prod P(\text{word}_i \mid \text{positive})
\]

---

## 🧠 Summary

| Aspect         | Description                             |
|----------------|-----------------------------------------|
| Type           | Supervised learning, Generative model   |
| Prediction     | Maximum a posteriori (MAP)              |
| Assumption     | Feature independence given the class    |
| Learning       | Closed-form (counting, estimating)      |

---

## 💡 Final Thought

Despite its simplicity and naive assumptions, **Naive Bayes** remains a **go-to algorithm** for many real-world problems — especially text-related tasks.

It’s often a great **baseline** model and is useful when **interpretability and speed** are essential.

