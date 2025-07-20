# 📍 Locally Weighted Regression (LWR) – A Flexible Non-Parametric Model

**Locally Weighted Regression (LWR)**, also known as **Locally Weighted Linear Regression (LWLR)** or **LOESS**, is a powerful non-parametric algorithm used to model complex relationships between variables by fitting **local models** instead of a single global function.

---

## 📚 What Is Locally Weighted Regression?

Unlike traditional linear regression, which fits **one global line** for all the data, **LWR fits a new line for each point of interest**, giving **more importance (weight)** to nearby data points and **less weight** to faraway ones.

> 💡 It's like zooming into the neighborhood of each point to make a highly localized prediction.

---

## 🔍 How Does It Work?

1. **Choose a query point** $x_q$ where you want to make a prediction.

2. Compute a **weight for each training point** based on its distance from $x_q$.  
   A commonly used function is the **Gaussian kernel**:

   $$
   w^{(i)} = \exp\left( - \frac{(x^{(i)} - x_q)^2}{2 \tau^2} \right)
   $$


   - $x^{(i)}$: the **i-th training example**
   - $x_q$: the **query point** — the input where you want to make a prediction
   - $\tau$ is the **bandwidth parameter** — it controls how "local" the model is.
   - Points closer to $x_q$ will have larger weights $w^{(i)}$.

3. Fit a **weighted linear regression** using the computed weights.

4. Predict the output:

   $$
   \hat{y}_q = \theta^\top x_q
   $$

   - Where $\theta$ is learned specifically for $x_q$ using the weighted training set.

5. Repeat for every new point you want to predict.

---

### 🔁 What happens with those weights?

Using these weights, the algorithm performs a **weighted linear regression** centered around each query point:

- It learns a **local line** (a local model) that fits only the neighborhood around $x_q$
- Then it **predicts** the value $\hat{y}_q$ at that point using the local model
- This process is **repeated** for every new input you want to predict


## ⚙️ What Is It Used For?

- Situations where the relationship between variables is **not globally linear**, but **locally linear**.
- When you want **high accuracy in local predictions**.
- Useful in:
  - **Time-series smoothing**
  - **Non-linear regression problems**
  - **Robust modeling of small datasets**

---

## ✅ Advantages

- ✔️ **Very flexible**: adapts to non-linear trends.
- ✔️ No need to assume a global model.
- ✔️ Works well for **small and medium datasets**.
- ✔️ Each region of the data can have its own behavior.

---

## ❌ Disadvantages

- ❌ **Slow prediction time**: must fit a model for each query.
- ❌ **Not scalable** for large datasets.
- ❌ Requires storing all training data in memory.
- ❌ Choosing the **right bandwidth (τ)** is critical.

---

## 🧠 Parametric vs Non-Parametric

| Feature                        | Parametric Regression     | Locally Weighted Regression       |
|-------------------------------|---------------------------|-----------------------------------|
| Assumes fixed model form      | ✅ Yes                    | ❌ No                              |
| Learns global parameters      | ✅                        | ❌ (fits locally for each point)   |
| Requires entire dataset to predict | ❌                    | ✅ Yes                             |
| Flexible with non-linear patterns | ❌ Limited             | ✅ Highly flexible                 |

---

## 📈 Visualization Example

Imagine a curve in a dataset.  
- **Linear regression** fits one straight line through it.  
- **LWR** fits many small lines locally, tracing the curve accurately.

---

## 🛠️ When Should You Use LWR?

- When your data shows **non-linear behavior**.
- When you want **smooth, localized predictions**.
- When the dataset is **small to moderate in size**.
- When **model interpretability and flexibility** are both important.

---

## 🧪 Summary

| Feature              | Value                                     |
|----------------------|-------------------------------------------|
| Algorithm type       | Non-parametric, supervised                |
| Prediction           | Local, per-query fitting                  |
| Input                | Training data, query point, bandwidth     |
| Output               | Local linear approximation                |
| Kernel               | Gaussian (commonly used)                  |

---

> 🔎 **Locally Weighted Regression is like tailoring a model for every new input**, making it extremely accurate in capturing local patterns, but less efficient in terms of computation.

---
