# White-Box Deep Learning: Mathematical Foundations of Backpropagation

## 🧠 Project Overview
In the era of high-level APIs, the core mathematical logic behind deep learning is often treated as a "black box." This project demystifies the backpropagation algorithm by mathematically deriving the gradients from scratch and comparing the results directly with the PyTorch Autograd engine.

This is particularly crucial for robotics and edge computing, where deploying models on resource-constrained hardware often requires translating high-level models into highly optimized, custom C/C++ matrix operations.

## ⚙️ Core Architecture & Math
The script models a simple linear layer and calculates the Mean Squared Error (MSE) loss. 

* **Forward Pass:** `Y_pred = X @ W`
* **Loss Function (MSE):** `L = 1/2 * sum((Y_pred - Y_target)^2)`
* **Backward Pass (Gradient w.r.t Weights):** Using multivariable calculus and matrix differentiation, the gradient is manually derived as:
  `dL/dW = X^T @ (Y_pred - Y_target)`

## 🚀 Execution & Verification
The Python script initializes random tensors with `float64` precision and compares the gradient matrices computed by:
1. Pure mathematical derivation (NumPy/Tensor dot products).
2. PyTorch's native `.backward()` autograd engine.

**Result:** The maximum absolute difference between the manual derivation and PyTorch's engine is exactly `0.0`, proving absolute mathematical parity.

## 🛠️ Environment
* Python 3.12 (Isolated Virtual Environment)
* PyTorch (CPU)
* NumPy
