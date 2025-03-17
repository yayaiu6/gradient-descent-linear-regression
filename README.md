# Gradient Descent for Linear Regression

##  Introduction
This project demonstrates how to implement **Linear Regression** using **Gradient Descent** to find the optimal parameters (`w` and `b`) for fitting a dataset.

---

##  Algorithm Overview
The goal is to determine the best parameters (`w`, `b`) that minimize the error between the actual values `y` and the predicted values `ŷ` using the equation:

ŷ = w * x + b

The error is measured using the **Mean Squared Error (MSE)** loss function:

Loss = (1/N) * Σ (y - ŷ)²

Where:
- `ŷ` is the predicted value.
- `N` is the total number of data points.

---

## Gradient Descent Formula
To minimize the loss, we update `w` and `b` using **partial derivatives**:

∂Loss/∂w = (-2/N) * Σ x * (y - (w * x + b))
∂Loss/∂b = (-2/N) * Σ (y - (w * x + b))


The parameters are updated as follows:

w = w - α * ∂Loss/∂w
b = b - α * ∂Loss/∂b

Where `α` (alpha) is the **learning rate**.

---

##  Code Implementation

This algorithm is implemented in Python using NumPy:

```python
# Gradient Descent for Linear Regression
# Install numpy if not installed

import numpy as np

# Initialize parameters
w = 0.0
b = 0.0

# Hyperparameters
learning_rate = 0.01

# Gradient Descent function
def descent(x, y, w, b, learning_rate):
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0]

    for xi, yi in zip(x, y):
        dldw += -2 * xi * (yi - (w * xi + b))
        dldb += -2 * (yi - (w * xi + b))

    # Update parameters
    w -= learning_rate * (dldw / N)
    b -= learning_rate * (dldb / N)

    return w, b

# Iteratively update weights
for epoch in range(1050):
    w, b = descent(x, y, w, b, learning_rate)
    yhat = w * x + b
    loss = np.mean((y - yhat) ** 2)
    
    print(f"Epoch {epoch}: Loss = {loss}, Weight = {w}, Bias = {b}")

 
``````
# Notes
You can change learning_rate and epochs to experiment with the results.
The dataset is randomly generated, so results may vary.
