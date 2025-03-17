# Gradient Descent for Linear Regression

##  Introduction
This project demonstrates how to implement **Linear Regression** using **Gradient Descent** to find the optimal parameters (`w` and `b`) for fitting a dataset.

---

##  Algorithm Overview
The goal is to determine the best parameters (`w`, `b`) that minimize the error between the actual values `y` and the predicted values `ŷ` using the equation:

\[
\hat{y} = w \cdot x + b
\]

The error is measured using the **Mean Squared Error (MSE)** loss function:

\[
\text{Loss} = \frac{1}{N} \sum (y - \hat{y})^2
\]

Where:
- `ŷ` is the predicted value.
- `N` is the total number of data points.

---

##  Gradient Descent Formula
To minimize the loss, we update `w` and `b` using **partial derivatives**:

\[
\frac{\partial \text{Loss}}{\partial w} = -\frac{2}{N} \sum x (y - (w \cdot x + b))
\]

\[
\frac{\partial \text{Loss}}{\partial b} = -\frac{2}{N} \sum (y - (w \cdot x + b))
\]

The parameters are updated as follows:

\[
w = w - \alpha \cdot \frac{\partial \text{Loss}}{\partial w}
\]

\[
b = b - \alpha \cdot \frac{\partial \text{Loss}}{\partial b}
\]

Where `α` (alpha) is the **learning rate**.

---

##  Code Implementation

This algorithm is implemented in Python using NumPy:

```python
import numpy as np

# Generate random dataset
x = np.random.randn(10, 1)
y = 2 * x + np.random.randn()

# Initialize parameters
w, b = 0.0, 0.0
learning_rate = 0.01

# Gradient Descent function
def descent(x, y, w, b, learning_rate):
    dldw, dldb = 0.0, 0.0
    N = x.shape[0]

    for xi, yi in zip(x, y):
        dldw += -2 * xi * (yi - (w * xi + b))
        dldb += -2 * (yi - (w * xi + b))

    w -= learning_rate * (1/N) * dldw
    b -= learning_rate * (1/N) * dldb
    return w, b

# Training loop
for epoch in range(1000):
    w, b = descent(x, y, w, b, learning_rate)
    yhat = w * x + b
    loss = np.sum((y - yhat) ** 2) / x.shape[0]
    print(f'Epoch {epoch}: Loss = {loss}, w = {w}, b = {b}')
``````
# Notes
You can change learning_rate and epochs to experiment with the results.
The dataset is randomly generated, so results may vary.
