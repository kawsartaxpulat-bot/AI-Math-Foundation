import torch

# ==========================================
# MangoBot Portfolio: Neural Network Foundation
# Demonstrating Math vs. PyTorch Autograd
# ==========================================

# 1. Initialize Input (X), Weights (W), and Target (Y)
# Using float64 for high mathematical precision
X = torch.randn(1, 3, dtype=torch.float64)        # 1x3 Input matrix
W = torch.randn(3, 2, requires_grad=True, dtype=torch.float64) # 3x2 Weight matrix
Y_target = torch.randn(1, 2, dtype=torch.float64) # 1x2 Target output

print("--- Forward Pass ---")
# Forward pass equation: Y_pred = X @ W
Y_pred = X @ W
print(f"Prediction:\n{Y_pred.detach().numpy()}")

# Loss function: Mean Squared Error (MSE)
# L = 1/2 * sum((Y_pred - Y_target)^2)
loss = 0.5 * torch.sum((Y_pred - Y_target)**2)
print(f"\nLoss: {loss.item():.4f}")

# ==========================================
# Method A: PyTorch Automatic Differentiation (Black Box)
# ==========================================
loss.backward()
pytorch_grad = W.grad.clone()

# ==========================================
# Method B: Pure Mathematical Derivation (White Box)
# ==========================================
# Based on Multivariable Calculus and Linear Algebra:
# dL/dW = X^T @ (Y_pred - Y_target)
manual_grad = X.t() @ (Y_pred - Y_target)

print("\n--- Backward Pass (Gradient Matrix dL/dW) ---")
print("1. PyTorch Autograd result:")
print(pytorch_grad.numpy())
print("\n2. Manual Mathematical Derivation result:")
print(manual_grad.detach().numpy())

# Verify absolute equality between Math and PyTorch
difference = torch.max(torch.abs(pytorch_grad - manual_grad)).item()
print(f"\nMaximum difference between Math and PyTorch: {difference}")
if difference < 1e-10:
    print("✅ SUCCESS: Mathematical derivation perfectly matches PyTorch Autograd engine.")
