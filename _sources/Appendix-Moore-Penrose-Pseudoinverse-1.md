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

# Case 1: Tall matrix (n > p)
X_tall = np.array([[1, 2], 
                   [3, 4], 
                   [5, 6]])  # 3x2
y = np.array([[1], [2], [3]])

# Form 1
w1 = np.linalg.inv(X_tall.T @ X_tall) @ X_tall.T @ y

# Form 2 (using pseudoinverse)
w2 = np.linalg.pinv(X_tall) @ y

print(f"Tall matrix solution: {w1.ravel()} = {w2.ravel()}")

# Case 2: Fat matrix (p > n)
X_fat = np.array([[1, 2, 3],
                  [4, 5, 6]])  # 2x3

# Form 2 (the only one that works)
w_fat = X_fat.T @ np.linalg.inv(X_fat @ X_fat.T) @ y

# Using general pseudoinverse
w_fat_pinv = np.linalg.pinv(X_fat) @ y

print(f"Fat matrix solution: {w_fat.ravel()} = {w_fat_pinv.ravel()}")
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
