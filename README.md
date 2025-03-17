Gradient Descent for Linear Regression
📌 Introduction
This project demonstrates how to implement Linear Regression using Gradient Descent to find the optimal parameters (w and b) for fitting a dataset.

🔍 Algorithm Overview
The goal is to determine the best parameters (w, b) that minimize the error between the actual values y and the predicted values ŷ using the equation:

𝑦
^
=
𝑤
⋅
𝑥
+
𝑏
y
^
​
 =w⋅x+b
The error is measured using the Mean Squared Error (MSE) loss function:

Loss
=
1
𝑁
∑
(
𝑦
−
𝑦
^
)
2
Loss= 
N
1
​
 ∑(y− 
y
^
​
 ) 
2
 
where N is the number of data points.

🛠️ How It Works
1️⃣ Data Initialization
Generate random dataset x and compute y using a linear equation with some noise.
Initialize parameters w = 0 and b = 0.
2️⃣ Gradient Descent Optimization
Gradient Descent updates the values of w and b by computing the partial derivatives of the loss function:

∂
Loss
∂
𝑤
=
−
2
𝑁
∑
𝑥
(
𝑦
−
(
𝑤
𝑥
+
𝑏
)
)
∂w
∂Loss
​
 =− 
N
2
​
 ∑x(y−(wx+b))
∂
Loss
∂
𝑏
=
−
2
𝑁
∑
(
𝑦
−
(
𝑤
𝑥
+
𝑏
)
)
∂b
∂Loss
​
 =− 
N
2
​
 ∑(y−(wx+b))
The parameters are updated using the following rules:

𝑤
=
𝑤
−
𝛼
⋅
∂
Loss
∂
𝑤
w=w−α⋅ 
∂w
∂Loss
​
 
𝑏
=
𝑏
−
𝛼
⋅
∂
Loss
∂
𝑏
b=b−α⋅ 
∂b
∂Loss
​
 
where α (alpha) is the learning rate.

💻 Code Implementation
This algorithm is implemented in Python using NumPy:

python
نسخ
تحرير
import numpy as np

# Generate random dataset
x = np.random.randn(10,1)
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
for epoch in range(1050):
    w, b = descent(x, y, w, b, learning_rate)
    yhat = w * x + b
    loss = np.sum((y - yhat) ** 2) / x.shape[0]
    print(f'Epoch {epoch}: Loss = {loss:.6f}, w = {w[0]}, b = {b}')
🚀 Key Features
✅ Implements Gradient Descent from scratch
✅ Uses NumPy for efficient numerical operations
✅ Demonstrates Linear Regression without relying on external ML libraries

📢 How to Use
Clone the repository:
sh
نسخ
تحرير
git clone https://github.com/YOUR_USERNAME/gradient-descent-linear-regression.git
cd gradient-descent-linear-regression
Run the script:
sh
نسخ
تحرير
python gradient_descent.py
📚 References
Gradient Descent - Wikipedia
Linear Regression - Stanford CS229
