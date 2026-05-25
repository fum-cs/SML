## Appendix: The Moore-Penrose Pseudoinverse 

### Part 1: Simple Definition of Pseudoinverse

#### What is the Pseudoinverse?

For a square invertible matrix $A$, the inverse $A^{-1}$ satisfies:

$$A A^{-1} = A^{-1} A = I$$

But for non-square or singular matrices, the ordinary inverse doesn't exist. The **pseudoinverse** $A^+$ (Moore-Penrose inverse) is a generalization that always exists and satisfies four special properties:

1. **$A A^+ A = A$** (reconstruction property)
2. **$A^+ A A^+ = A^+$** (idempotence)
3. **$(A A^+)^T = A A^+$** (symmetry)
4. **$(A^+ A)^T = A^+ A$** (symmetry)

For machine learning, the most important property is:
- If $A$ is $m \times n$, then $A^+$ is $n \times m$
- $A^+$ gives the "best possible" solution to $A\mathbf{x} = \mathbf{b}$ for any $A$

---

### Part 2: Two Forms of Pseudoinverse for OLS

#### The OLS Problem

We want to solve:

$$X \mathbf{w} = \mathbf{y}$$

where $X$ is $n \times p$ ($n$ samples, $p$ features).

The pseudoinverse solution is:

$$\mathbf{w}^* = X^+ \mathbf{y}$$

Depending on the shape of $X$, $X^+$ takes different forms.

---

#### Form 1: Tall Matrix ($n > p$, full column rank)

**Derivation (from your course):**

We minimize $\|X\mathbf{w} - \mathbf{y}\|^2$:

$$\nabla_{\mathbf{w}} \|X\mathbf{w} - \mathbf{y}\|^2 = 2X^T X \mathbf{w} - 2X^T \mathbf{y} = 0$$

$$X^T X \mathbf{w} = X^T \mathbf{y}$$

Since $n > p$ and $X$ has full column rank, $X^T X$ is $p \times p$ and invertible:

$$\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$$

Therefore:

$$X^+ = (X^T X)^{-1} X^T$$

---

#### Form 2: Fat Matrix ($p > n$, full row rank)



**When $p > n$** (more features than samples), $X\mathbf{w} = \mathbf{y}$ has infinitely many solutions. We want the **minimum norm solution**:

$$\min \|\mathbf{w}\|^2 \quad \text{subject to} \quad X\mathbf{w} = \mathbf{y}$$

**Key fact:** The minimum norm solution lies in the row space of $X$, so we can write $\mathbf{w} = X^T \boldsymbol{\alpha}$ for some $\boldsymbol{\alpha} \in \mathbb{R}^n$.

**Substitute into the constraint:**

$$X (X^T \boldsymbol{\alpha}) = \mathbf{y}$$

$$(X X^T) \boldsymbol{\alpha} = \mathbf{y}$$

Since $X X^T$ is $n \times n$ and invertible (assuming $X$ has full row rank):

$$\boldsymbol{\alpha} = (X X^T)^{-1} \mathbf{y}$$

**Therefore:**

$$\mathbf{w} = X^T \boldsymbol{\alpha} = X^T (X X^T)^{-1} \mathbf{y}$$

**Thus:**

$$X^+ = X^T (X X^T)^{-1}$$

---

#### Alternative Derivation Using Optimization Constraint

**Another way to derive Form 2 (minimum norm interpretation):**

For fat matrices ($p > n$), there are infinitely many solutions to $X\mathbf{w} = \mathbf{y}$. We want the **minimum norm solution**:

$$\min_{\mathbf{w}} \|\mathbf{w}\|^2 \quad \text{subject to} \quad X\mathbf{w} = \mathbf{y}$$

Using Lagrange multipliers:

$$L(\mathbf{w}, \boldsymbol{\lambda}) = \frac{1}{2}\mathbf{w}^T \mathbf{w} + \boldsymbol{\lambda}^T (\mathbf{y} - X\mathbf{w})$$

Gradient with respect to $\mathbf{w}$:

$$\frac{\partial L}{\partial \mathbf{w}} = \mathbf{w} - X^T \boldsymbol{\lambda} = 0$$

So:

$$\mathbf{w} = X^T \boldsymbol{\lambda}$$

Multiply by $X$:

$$X\mathbf{w} = X X^T \boldsymbol{\lambda} = \mathbf{y}$$

Since $X X^T$ is invertible:

$$\boldsymbol{\lambda} = (X X^T)^{-1} \mathbf{y}$$

Therefore:

$$\mathbf{w} = X^T (X X^T)^{-1} \mathbf{y}$$

Again we get:

$$X^+ = X^T (X X^T)^{-1}$$

---

### Part 3: Verification That Both Forms Satisfy Pseudoinverse Properties

#### For Tall Matrix: $X^+ = (X^T X)^{-1} X^T$

Check property 1: $X X^+ X = X$

$$X X^+ X = X (X^T X)^{-1} X^T X = X I = X \quad \checkmark$$

Check property 2: $X^+ X X^+ = X^+$

$$X^+ X X^+ = (X^T X)^{-1} X^T X (X^T X)^{-1} X^T = (X^T X)^{-1} X^T = X^+ \quad \checkmark$$

#### For Fat Matrix: $X^+ = X^T (X X^T)^{-1}$

Check property 1: $X X^+ X = X$

$$X X^+ X = X X^T (X X^T)^{-1} X = I X = X \quad \checkmark$$

Check property 2: $X^+ X X^+ = X^+$

$$X^+ X X^+ = X^T (X X^T)^{-1} X X^T (X X^T)^{-1} = X^T (X X^T)^{-1} = X^+ \quad \checkmark$$

---

### Part 4: Connection Between Both Forms

**They are related through the SVD:**

Let $X = U \Sigma V^T$ where:
- $U$ is $n \times n$ orthogonal
- $\Sigma$ is $n \times p$ diagonal (singular values)
- $V$ is $p \times p$ orthogonal

Then:
$$X^+ = V \Sigma^+ U^T$$

where $\Sigma^+$ is the transpose of $\Sigma$ with reciprocals of non-zero singular values.

- For tall $X$ ($n > p$): $\Sigma^+$ is $p \times n$
- For fat $X$ ($p > n$): $\Sigma^+$ is $p \times n$ as well

Both formulas are special cases of this unified SVD definition.

---

### Part 5: Summary Table with Derivations

| Case | Condition | Formula | Derivation Method |
|------|-----------|---------|-------------------|
| **Tall** | $n > p$, full column rank | $X^+ = (X^T X)^{-1} X^T$ | Set derivative to zero, solve |
| **Fat** | $p > n$, full row rank | $X^+ = X^T (X X^T)^{-1}$ | Substitute $\mathbf{w} = X^T \mathbf{u}$ OR Lagrange multipliers |
| **Square invertible** | $n = p$, full rank | $X^+ = X^{-1}$ | Special case of both |
| **General** | Any shape/rank | $X^+ = V \Sigma^+ U^T$ | SVD |

---

### Part 6: Numerical Example

```python
import numpy as np

print("="*60)
print("PSEUDOINVERSE DEMONSTRATION")
print("="*60)

# Case 1: Tall matrix (more samples than features)
print("\n📊 CASE 1: Tall Matrix (n > p)")
print("-"*40)
X_tall = np.array([[1, 2], 
                   [3, 4], 
                   [5, 6]])  # Shape: (3, 2) - 3 samples, 2 features
y_tall = np.array([1, 2, 3])  # Shape: (3,) - 3 targets

print(f"X shape: {X_tall.shape}")
print(f"y shape: {y_tall.shape}")
print(f"Problem: {X_tall.shape[0]} equations, {X_tall.shape[1]} unknowns")

# Solution using normal equations (tall matrix formula)
w_tall = np.linalg.inv(X_tall.T @ X_tall) @ X_tall.T @ y_tall
print(f"\nSolution (minimizes ||Xw - y||²): {w_tall}")

# Using numpy's pinv
w_tall_pinv = np.linalg.pinv(X_tall) @ y_tall
print(f"Using np.linalg.pinv: {w_tall_pinv}")

# Verify
print(f"\nPrediction: Xw = {X_tall @ w_tall}")
print(f"Target y:    {y_tall}")
print(f"MSE: {np.mean((X_tall @ w_tall - y_tall)**2):.6f}")

# Case 2: Fat matrix (more features than samples)
print("\n" + "="*60)
print("📊 CASE 2: Fat Matrix (p > n)")
print("-"*40)
X_fat = np.array([[1, 2, 3],
                  [4, 5, 6]])  # Shape: (2, 3) - 2 samples, 3 features
y_fat = np.array([1, 2])  # Shape: (2,) - 2 targets

print(f"X shape: {X_fat.shape}")
print(f"y shape: {y_fat.shape}")
print(f"Problem: {X_fat.shape[0]} equations, {X_fat.shape[1]} unknowns")
print("→ Infinite solutions exist. We want the minimum norm solution.")

# Solution using fat matrix formula (minimum norm)
XXT = X_fat @ X_fat.T  # Shape: (2, 2)
inv_XXT = np.linalg.inv(XXT)
w_fat = X_fat.T @ inv_XXT @ y_fat

print(f"\nMinimum norm solution: {w_fat}")
print(f"Norm of w: {np.linalg.norm(w_fat):.4f}")

# Using numpy's pinv
w_fat_pinv = np.linalg.pinv(X_fat) @ y_fat
print(f"Using np.linalg.pinv: {w_fat_pinv}")

# Verify solution satisfies Xw = y
print(f"\nVerification:")
print(f"Xw = {X_fat @ w_fat}")
print(f"y  = {y_fat}")

# Show that other solutions exist with larger norm
print(f"\n📐 Other possible solutions:")
# One particular solution (not minimum norm)
w_other = np.array([0, 0.5, 0])  # This also satisfies Xw = y?
print(f"Alternative w = {w_other}")
print(f"Norm = {np.linalg.norm(w_other):.4f} (larger than {np.linalg.norm(w_fat):.4f})")
print(f"Xw = {X_fat @ w_other}")

# Case 3: Square invertible matrix
print("\n" + "="*60)
print("📊 CASE 3: Square Invertible Matrix (n = p)")
print("-"*40)
X_square = np.array([[1, 2],
                     [3, 4]])  # Shape: (2, 2)
y_square = np.array([1, 2])

print(f"X shape: {X_square.shape}")
print(f"X is square and invertible (det = {np.linalg.det(X_square):.1f})")

# Standard inverse
w_inv = np.linalg.inv(X_square) @ y_square

# Pseudoinverse (should give same result)
w_pinv = np.linalg.pinv(X_square) @ y_square

print(f"\nUsing inverse: {w_inv}")
print(f"Using pinv:    {w_pinv}")
print(f"Are they equal? {np.allclose(w_inv, w_pinv)}")

print("\n" + "="*60)
print("💡 KEY INSIGHTS:")
print("="*60)
print("1. Tall matrix (n > p): Unique solution minimizing MSE")
print("2. Fat matrix (p > n): Infinite solutions → choose minimum norm")
print("3. Square matrix: Pseudoinverse = inverse")
print("4. np.linalg.pinv() handles all cases automatically using SVD")
```

```
============================================================
PSEUDOINVERSE DEMONSTRATION
============================================================

📊 CASE 1: Tall Matrix (n > p)
----------------------------------------
X shape: (3, 2)
y shape: (3,)
Problem: 3 equations, 2 unknowns

Solution (minimizes ||Xw - y||²): [4.4408921e-16 5.0000000e-01]
Using np.linalg.pinv: [0.  0.5]

Prediction: Xw = [1. 2. 3.]
Target y:    [1 2 3]
MSE: 0.000000

============================================================
📊 CASE 2: Fat Matrix (p > n)
----------------------------------------
X shape: (2, 3)
y shape: (2,)
Problem: 2 equations, 3 unknowns
→ Infinite solutions exist. We want the minimum norm solution.

Minimum norm solution: [-0.05555556  0.11111111  0.27777778]
Norm of w: 0.3043
Using np.linalg.pinv: [-0.05555556  0.11111111  0.27777778]

Verification:
Xw = [1. 2.]
y  = [1 2]

📐 Other possible solutions:
Alternative w = [0.  0.5 0. ]
Norm = 0.5000 (larger than 0.3043)
Xw = [1.  2.5]

============================================================
📊 CASE 3: Square Invertible Matrix (n = p)
----------------------------------------
X shape: (2, 2)
X is square and invertible (det = -2.0)

Using inverse: [0.  0.5]
Using pinv:    [-4.4408921e-16  5.0000000e-01]
Are they equal? True

============================================================
💡 KEY INSIGHTS:
============================================================
1. Tall matrix (n > p): Unique solution minimizing MSE
2. Fat matrix (p > n): Infinite solutions → choose minimum norm
3. Square matrix: Pseudoinverse = inverse
4. np.linalg.pinv() handles all cases automatically using SVD
```


---

### Part 7: Key Takeaway

The OLS closed-form solution $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$:
- **Assumes $n \ge p$** and $X$ has full column rank
- **Derivation:** Set gradient to zero, solve assuming invertibility

The alternative form $\mathbf{w} = X^T (X X^T)^{-1} \mathbf{y}$:
- **Works when $p > n$** and $X$ has full row rank
- **Derivation:** Substitute $\mathbf{w} = X^T \mathbf{u}$ or use Lagrange multipliers for minimum norm

**Both are special cases of the pseudoinverse $X^+$, which always exists and can be computed via SVD.**

This is why, in practice, we often write:
$$\mathbf{w}^* = X^+ \mathbf{y}$$

without worrying about the shape of $X$!

---
