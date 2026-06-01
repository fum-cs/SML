## Appendix: Direct Maximization of the GMM Log-Likelihood

**Direct MLE for a Univariate Gaussian Mixture Model**

### 1. Problem Setup

We extend the univariate Gaussian case from Section 3 of the MLE notes. Now, we assume the data $\mathcal{D} = \{x_1, x_2, \dots, x_n\}$ are drawn from a **mixture** of two univariate Gaussian distributions:

$$
p(x | \theta) = \pi_1 \, \mathcal{N}(x | \mu_1, \sigma_1^2) + \pi_2 \, \mathcal{N}(x | \mu_2, \sigma_2^2)
$$

where:
- $\mathcal{N}(x | \mu_i, \sigma_i^2) = \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left(-\frac{(x - \mu_i)^2}{2\sigma_i^2}\right)$
- $\pi_1 + \pi_2 = 1$, $\pi_i \ge 0$ are the mixing coefficients (prior probabilities of each component).
- $\theta = \{\pi_1, \mu_1, \sigma_1^2, \mu_2, \sigma_2^2\}$.

We have $n$ i.i.d. samples. The goal is to find $\hat{\theta}$ that maximizes the likelihood $L(\theta|\mathcal{D})$.

### 2. Likelihood and Log-Likelihood Functions

The likelihood function is:

$$
L(\theta|\mathcal{D}) = \prod_{j=1}^n \left[ \pi_1 \mathcal{N}(x_j | \mu_1, \sigma_1^2) + \pi_2 \mathcal{N}(x_j | \mu_2, \sigma_2^2) \right]
$$

The log-likelihood is:

$$
\ell(\theta) = \sum_{j=1}^n \ln \left( \pi_1 \mathcal{N}(x_j | \mu_1, \sigma_1^2) + \pi_2 \mathcal{N}(x_j | \mu_2, \sigma_2^2) \right)
$$

Unlike the single Gaussian case (Section 3.3), the **log of a sum** prevents us from separating the terms into a simple additive form.

### 3. Attempting Direct Maximization

We try to maximize $\ell(\theta)$ by setting partial derivatives to zero.

#### 3.1 Derivative with respect to $\mu_1$

$$
\frac{\partial \ell}{\partial \mu_1} = \sum_{j=1}^n \frac{1}{\pi_1 \mathcal{N}(x_j|\mu_1,\sigma_1^2) + \pi_2 \mathcal{N}(x_j|\mu_2,\sigma_2^2)} \cdot \pi_1 \frac{\partial \mathcal{N}(x_j|\mu_1,\sigma_1^2)}{\partial \mu_1}
$$

We know from the standard [MLE derivation](03-MLE-intro.md):

$$
\frac{\partial \mathcal{N}(x_j|\mu_1,\sigma_1^2)}{\partial \mu_1} = \mathcal{N}(x_j|\mu_1,\sigma_1^2) \cdot \frac{(x_j - \mu_1)}{\sigma_1^2}
$$

Therefore:

$$
\frac{\partial \ell}{\partial \mu_1} = \sum_{j=1}^n \frac{\pi_1 \mathcal{N}(x_j|\mu_1,\sigma_1^2)}{\pi_1 \mathcal{N}(x_j|\mu_1,\sigma_1^2) + \pi_2 \mathcal{N}(x_j|\mu_2,\sigma_2^2)} \cdot \frac{(x_j - \mu_1)}{\sigma_1^2}
$$

Define the **responsibility** (posterior probability) of component 1 for point $x_j$:

$$
w_{1j} = P(C_1 | x_j, \theta) = \frac{\pi_1 \mathcal{N}(x_j|\mu_1,\sigma_1^2)}{\pi_1 \mathcal{N}(x_j|\mu_1,\sigma_1^2) + \pi_2 \mathcal{N}(x_j|\mu_2,\sigma_2^2)}
$$

Then:

$$
\frac{\partial \ell}{\partial \mu_1} = \sum_{j=1}^n w_{1j} \frac{(x_j - \mu_1)}{\sigma_1^2} = \frac{1}{\sigma_1^2} \sum_{j=1}^n w_{1j} (x_j - \mu_1)
$$

Setting to zero:

$$
\sum_{j=1}^n w_{1j} (x_j - \mu_1) = 0 \quad \Rightarrow \quad \boxed{\mu_1 = \frac{\sum_{j=1}^n w_{1j} x_j}{\sum_{j=1}^n w_{1j}}}
$$

#### 3.2 Derivative with respect to $\mu_2$

By symmetry:

$$
\boxed{\mu_2 = \frac{\sum_{j=1}^n w_{2j} x_j}{\sum_{j=1}^n w_{2j}}}
$$

where $w_{2j} = 1 - w_{1j}$.

#### 3.3 Derivative with respect to $\sigma_1^2$

Let $\tau_1 = \sigma_1^2$. Using $\frac{\partial \ln \mathcal{N}}{\partial \tau_1} = -\frac{1}{2\tau_1} + \frac{(x_j - \mu_1)^2}{2\tau_1^2}$:

$$
\frac{\partial \ell}{\partial \tau_1} = \sum_{j=1}^n w_{1j} \left( -\frac{1}{2\tau_1} + \frac{(x_j - \mu_1)^2}{2\tau_1^2} \right) = 0
$$

Multiply by $2\tau_1^2$:

$$
\sum_{j=1}^n w_{1j} \left( -\tau_1 + (x_j - \mu_1)^2 \right) = 0
$$

$$
\Rightarrow \boxed{\sigma_1^2 = \frac{\sum_{j=1}^n w_{1j} (x_j - \mu_1)^2}{\sum_{j=1}^n w_{1j}}}
$$

#### 3.4 Derivative with respect to $\sigma_2^2$

By symmetry:

$$
\boxed{\sigma_2^2 = \frac{\sum_{j=1}^n w_{2j} (x_j - \mu_2)^2}{\sum_{j=1}^n w_{2j}}}
$$

#### 3.5 Derivative with respect to $\pi_1$ (with $\pi_2 = 1 - \pi_1$)

We have:

$$
\frac{\partial \ell}{\partial \pi_1} = \sum_{j=1}^n \frac{\mathcal{N}(x_j|\mu_1,\sigma_1^2) - \mathcal{N}(x_j|\mu_2,\sigma_2^2)}{\pi_1 \mathcal{N}(x_j|\mu_1,\sigma_1^2) + \pi_2 \mathcal{N}(x_j|\mu_2,\sigma_2^2)} = 0
$$

Rewrite using $w_{1j}$ and $w_{2j}$:

$$
\frac{\mathcal{N}(x_j|\mu_1,\sigma_1^2)}{\pi_1 \mathcal{N}(\cdot) + \pi_2 \mathcal{N}(\cdot)} = \frac{w_{1j}}{\pi_1}, \quad \frac{\mathcal{N}(x_j|\mu_2,\sigma_2^2)}{\pi_1 \mathcal{N}(\cdot) + \pi_2 \mathcal{N}(\cdot)} = \frac{w_{2j}}{\pi_2}
$$

Thus:

$$
\sum_{j=1}^n \left( \frac{w_{1j}}{\pi_1} - \frac{w_{2j}}{\pi_2} \right) = 0
$$

Let $N_1 = \sum_j w_{1j}$, $N_2 = \sum_j w_{2j} = n - N_1$. Then:

$$
\frac{N_1}{\pi_1} - \frac{N_2}{1 - \pi_1} = 0 \quad \Rightarrow \quad N_1(1 - \pi_1) = N_2 \pi_1
$$

$$
N_1 = \pi_1 (N_1 + N_2) = \pi_1 n \quad \Rightarrow \quad \boxed{\pi_1 = \frac{N_1}{n}, \quad \pi_2 = \frac{N_2}{n}}
$$

### 4. The Fundamental Problem: A Coupled System

Collecting the direct MLE conditions:

$$
\boxed{
\begin{aligned}
\pi_i &= \frac{1}{n} \sum_{j=1}^n w_{ij} \\[1ex]
\mu_i &= \frac{\sum_{j=1}^n w_{ij} x_j}{\sum_{j=1}^n w_{ij}} \\[1ex]
\sigma_i^2 &= \frac{\sum_{j=1}^n w_{ij} (x_j - \mu_i)^2}{\sum_{j=1}^n w_{ij}}
\end{aligned}
}
$$

where

$$
w_{ij} = \frac{\pi_i \mathcal{N}(x_j | \mu_i, \sigma_i^2)}{\pi_1 \mathcal{N}(x_j | \mu_1, \sigma_1^2) + \pi_2 \mathcal{N}(x_j | \mu_2, \sigma_2^2)}.
$$

**Crucial observation:** The responsibilities $w_{ij}$ depend on **all** the parameters $\{\pi_i, \mu_i, \sigma_i^2\}$, while the update equations for $\pi_i, \mu_i, \sigma_i^2$ depend on the $w_{ij}$.

This is a **system of coupled fixed-point equations**, not a closed-form solution. Unlike [the single Gaussian case](03-MLE-intro.md), we cannot solve for each parameter independently.

### 5. Why the EM Algorithm is Needed

| Single Gaussian (MLE) | Gaussian Mixture (Direct MLE) |
|----------------------|-------------------------------|
| $\frac{\partial \ell}{\partial \mu} = 0$ gives $\mu = \frac{1}{n}\sum x_j$ directly | $\frac{\partial \ell}{\partial \mu_1} = 0$ gives $\mu_1 = \frac{\sum w_{1j} x_j}{\sum w_{1j}}$, but $w_{1j}$ depends on $\mu_1$ |
| Parameters decouple | Parameters couple through responsibilities |
| Closed-form solution exists | No closed-form solution |

The EM algorithm (Expectation-Maximization) provides an iterative procedure to solve these fixed-point equations:

1. **E-step:** Using current parameter estimates $\theta^{(t)}$, compute $w_{ij}^{(t)}$ (the posterior probabilities).
2. **M-step:** Treating $w_{ij}^{(t)}$ as fixed, update $\pi_i^{(t+1)}, \mu_i^{(t+1)}, \sigma_i^{2,(t+1)}$ using the equations above.

Each iteration increases the log-likelihood, and the algorithm converges to a local maximum. This is precisely the bridge mentioned in Section 7 of your MLE notes: "Gaussian Mixture Models (GMM): Extends single Gaussians to mixtures. MLE is used, but requires the EM Algorithm."

### 6. Summary

- Direct differentiation of the mixture log-likelihood yields update equations that **look like** weighted MLE formulas.
- However, the weights $w_{ij}$ are themselves functions of the unknown parameters.
- This creates a **coupled system** with no closed-form solution.
- **EM algorithm** solves this system iteratively: fix weights (E-step), update parameters (M-step), repeat.

Thus, while the single Gaussian enjoys a simple closed-form MLE (sample mean and variance), the Gaussian mixture requires the iterative EM approach—a natural extension of the MLE principle to more complex, latent-variable models.