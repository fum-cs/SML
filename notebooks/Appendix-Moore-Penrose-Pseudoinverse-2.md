
## Why the Minimum Norm Solution Lies in the Row Space of $X$

### Part 1: Understanding the Four Fundamental Subspaces

For any matrix $X$ of size $n \times p$:

| Subspace | Definition | Dimension | Relationship |
|----------|------------|-----------|--------------|
| **Row space** $R(X)$ | All linear combinations of rows of $X$ | rank($X$) = $r$ | In $\mathbb{R}^p$ |
| **Null space** $N(X)$ | Vectors $\mathbf{w}$ such that $X\mathbf{w} = 0$ | $p - r$ | In $\mathbb{R}^p$ |
| **Column space** $C(X)$ | All linear combinations of columns of $X$ | rank($X$) = $r$ | In $\mathbb{R}^n$ |
| **Left null space** $N(X^T)$ | Vectors $\mathbf{y}$ such that $X^T\mathbf{y} = 0$ | $n - r$ | In $\mathbb{R}^n$ |

**Key relationship:** Row space is orthogonal to null space:
- Any vector in the row space is perpendicular to any vector in the null space
- $\mathbb{R}^p = \text{Row Space} \oplus \text{Null Space}$ (direct sum)

---

### Part 2: What Does $X\mathbf{w} = \mathbf{y}$ Mean Geometrically?

The equation $X\mathbf{w} = \mathbf{y}$ means:
- $\mathbf{y}$ is a linear combination of the **columns** of $X$
- $\mathbf{w}$ contains the coefficients for this combination

**Example:**
$$X = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \quad \mathbf{w} = \begin{bmatrix} w_1 \\ w_2 \\ w_3 \end{bmatrix}$$

Then:
$$X\mathbf{w} = w_1\begin{bmatrix} 1 \\ 4 \end{bmatrix} + w_2\begin{bmatrix} 2 \\ 5 \end{bmatrix} + w_3\begin{bmatrix} 3 \\ 6 \end{bmatrix} = \mathbf{y}$$

---

### Part 3: Why Are There Multiple Solutions When $p > n$?

When $p > n$ (more unknowns than equations), the null space contains non-zero vectors.

**Example:** Let's find the null space of a simple $1 \times 3$ matrix:

$$X = \begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$$

Solve $X\mathbf{w} = 0$:
$$w_1 + 2w_2 + 3w_3 = 0$$

This has infinitely many solutions:
$$\mathbf{w} = \begin{bmatrix} -2w_2 -3w_3 \\ w_2 \\ w_3 \end{bmatrix} = w_2\begin{bmatrix} -2 \\ 1 \\ 0 \end{bmatrix} + w_3\begin{bmatrix} -3 \\ 0 \\ 1 \end{bmatrix}$$

The null space is 2-dimensional.

---

### Part 4: The Key Insight - Decomposing Any Solution

**Any vector $\mathbf{w} \in \mathbb{R}^p$ can be uniquely decomposed as:**

$$\mathbf{w} = \mathbf{w}_r + \mathbf{w}_n$$

where:
- $\mathbf{w}_r$ is in the **row space** of $X$
- $\mathbf{w}_n$ is in the **null space** of $X$
- $\mathbf{w}_r \perp \mathbf{w}_n$ (orthogonal)

**Why is this decomposition useful?** Because:

$$X\mathbf{w} = X(\mathbf{w}_r + \mathbf{w}_n) = X\mathbf{w}_r + X\mathbf{w}_n = X\mathbf{w}_r + 0 = X\mathbf{w}_r$$

**The null space component contributes nothing to $X\mathbf{w}$!**

---

### Part 5: Visualizing the Decomposition

Think of $\mathbb{R}^p$ as a 3D space:

```
                    ▲
                    |  Null Space
                    |  (line/plane through origin)
                    |
        w_n         |
        ↑           |
        |           |
        |           |
    w_r |           |
    ←---•--------→  |  Row Space (orthogonal to Null Space)
        |           |
        |           |
        |           |
```

- Row space: a line (1D) or plane (2D) through origin
- Null space: perpendicular line or plane
- **Any vector can be split into row space part + null space part**

---

### Part 6: Why the Minimum Norm Solution Uses Row Space

**Norm squared:**
$$\|\mathbf{w}\|^2 = \|\mathbf{w}_r\|^2 + \|\mathbf{w}_n\|^2$$

Since $\mathbf{w}_r \perp \mathbf{w}_n$ (Pythagorean theorem).

**To minimize $\|\mathbf{w}\|$ given $X\mathbf{w} = \mathbf{y}$:**

Since $X\mathbf{w}_n = 0$, we have:
$$X\mathbf{w} = X(\mathbf{w}_r + \mathbf{w}_n) = X\mathbf{w}_r = \mathbf{y}$$

The constraint only involves $\mathbf{w}_r$! 
The null space component $\mathbf{w}_n$ can be **anything** and doesn't affect $X\mathbf{w}$.

Therefore:
- Choose $\mathbf{w}_n = 0$ to minimize the norm
- Then $\mathbf{w} = \mathbf{w}_r$ (pure row space vector)

**Conclusion:** The minimum norm solution has no null space component → lies entirely in the row space.

---

### Part 7: Why Can We Write $\mathbf{w} = X^T \boldsymbol{\alpha}$?

**Theorem:** The row space of $X$ equals the column space of $X^T$.

- Row space of $X$ = span of rows of $X$
- Column space of $X^T$ = span of columns of $X^T$

But columns of $X^T$ are the **rows of $X$**! So they're the same space.

**Therefore:** Any vector $\mathbf{w}$ in the row space can be written as a linear combination of columns of $X^T$.

Let the columns of $X^T$ be $\mathbf{c}_1, \mathbf{c}_2, ..., \mathbf{c}_n$ (each is $p \times 1$).

Then:
$$\mathbf{w} = \alpha_1 \mathbf{c}_1 + \alpha_2 \mathbf{c}_2 + ... + \alpha_n \mathbf{c}_n$$

But this is exactly the matrix multiplication $X^T \boldsymbol{\alpha}$, where:
$$\boldsymbol{\alpha} = \begin{bmatrix} \alpha_1 \\ \alpha_2 \\ \vdots \\ \alpha_n \end{bmatrix}$$

---

### Part 8: Concrete Example

Let $X = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$ (size $2 \times 3$)

**Row space vectors:** Linear combinations of $(1,2,3)$ and $(4,5,6)$

**$X^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$** (size $3 \times 2$)

Columns of $X^T$:
- Column 1: $\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ (first row of $X$)
- Column 2: $\begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}$ (second row of $X$)

Any $\mathbf{w} = X^T \boldsymbol{\alpha} = \alpha_1\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} + \alpha_2\begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}$

This is exactly a linear combination of the rows of $X$ (written as column vectors).

**Thus $\mathbf{w}$ lies in the row space!**

---

### Part 9: Numerical Illustration

```python
import numpy as np

# Fat matrix: 2 samples, 3 features
X = np.array([[1, 2, 3],
              [4, 5, 6]])

# Some target
y = np.array([[7], [8]])

# Step 1: Find a particular solution (not minimum norm)
w_particular = np.linalg.pinv(X) @ y  # This gives minimum norm

# Step 2: Show that any solution = w_row + w_null
# Find basis for null space
from scipy.linalg import null_space
null_basis = null_space(X)  # 3x1 vector

# Generate a random null space vector
w_null = null_basis * 3.14

# This will also satisfy Xw = y
w_another = w_particular + w_null
print(f"X @ w_particular = {X @ w_particular.ravel()}")
print(f"X @ w_another = {X @ w_another.ravel()}")
print(f"Same? {np.allclose(X @ w_particular, X @ w_another)}")

# Compute norms
print(f"Norm of w_particular (row space): {np.linalg.norm(w_particular):.3f}")
print(f"Norm of w_another: {np.linalg.norm(w_another):.3f}")
print(f"w_particular has smaller norm!")
```

---

### Part 10: Summary Table

| Property | Explanation |
|----------|-------------|
| **Row space vectors** | Can be written as $X^T \boldsymbol{\alpha}$ |
| **Null space vectors** | Satisfy $X\mathbf{w}_n = 0$ |
| **Any solution** | $\mathbf{w} = \mathbf{w}_r + \mathbf{w}_n$ |
| **Effect on $X\mathbf{w}$** | $X\mathbf{w} = X\mathbf{w}_r$ (null part disappears) |
| **Minimum norm** | Set $\mathbf{w}_n = 0$ → $\mathbf{w} = \mathbf{w}_r$ |
| **Therefore** | $\mathbf{w}_{\min} = X^T \boldsymbol{\alpha}$ for some $\boldsymbol{\alpha}$ |

---

### Part 11: Intuitive Analogy

Imagine you're trying to hit a target $\mathbf{y}$ by shooting from different positions:

- **Column space** = all reachable targets
- **Row space** = "fire control" parameters that actually matter
- **Null space** = adjustments that don't change where you hit

If you want to use minimum ammunition (small $\|\mathbf{w}\|$), you make only the necessary adjustments (row space) and avoid wasted moves (null space).

**Thus:** $\mathbf{w} = X^T \boldsymbol{\alpha}$ means "only make adjustments that actually affect the target."

---

### Part 12: The Mathematical Proof

**Theorem:** $R(X^T) = \{\mathbf{w} \in \mathbb{R}^p : \mathbf{w} = X^T \boldsymbol{\alpha} \text{ for some } \boldsymbol{\alpha} \in \mathbb{R}^n\}$

**Proof:**
- $X^T \boldsymbol{\alpha} = \alpha_1 \mathbf{c}_1 + ... + \alpha_n \mathbf{c}_n$ where $\mathbf{c}_i$ are columns of $X^T$
- But columns of $X^T$ are rows of $X$
- Thus any linear combination of columns of $X^T$ is a linear combination of rows of $X$
- Row space of $X$ = span(rows of $X$) = span(columns of $X^T$) = column space of $X^T = \{X^T \boldsymbol{\alpha} : \boldsymbol{\alpha} \in \mathbb{R}^n\}$

**Therefore:** The minimum norm solution $\mathbf{w}^*$ (which lies in row space) can be written as $\mathbf{w}^* = X^T \boldsymbol{\alpha}^*$ for some $\boldsymbol{\alpha}^*$.

---

### Conclusion

The "key fact" is a fundamental theorem of linear algebra:
- $\text{Row space} = \text{Column space of } X^T = \{X^T \boldsymbol{\alpha} : \boldsymbol{\alpha} \in \mathbb{R}^n\}$
- Minimum norm solution has no null space component → lies in row space → can be written as $X^T \boldsymbol{\alpha}$

