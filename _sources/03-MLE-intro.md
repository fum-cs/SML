# Maximum Likelihood Estimation

In this simplified explanation, we focus on estimating the parameters of a univariate Gaussian distribution using Maximum Likelihood Estimation (MLE). The univariate Gaussian distribution is defined as:

$$P(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

where:
- $\mu$ is the mean,
- $\sigma^2$ is the variance.

Given a dataset $\mathcal{D} = \{x_1, x_2, \dots, x_n\}$ of $n$ i.i.d. samples, the goal is to estimate $\mu$ and $\sigma^2$ by maximizing the likelihood function.

## Likelihood Function

The likelihood function is the probability of observing the dataset $\mathcal{D}$ given the parameters $\mu$ and $\sigma^2$:

$$L(\mu, \sigma^2) = \prod_{k=1}^n P(x_k | \mu, \sigma^2) = \prod_{k=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_k - \mu)^2}{2\sigma^2}\right)$$

## Log-Likelihood Function

To simplify the computation, we take the logarithm of the likelihood function (log-likelihood):

$$\ell(\mu, \sigma^2) = \log L(\mu, \sigma^2) = \sum_{k=1}^n \left[ -\frac{1}{2} \log(2\pi\sigma^2) - \frac{(x_k - \mu)^2}{2\sigma^2} \right]$$

Simplifying further:

$$\ell(\mu, \sigma^2) = -\frac{n}{2} \log(2\pi) - \frac{n}{2} \log(\sigma^2) - \frac{1}{2\sigma^2} \sum_{k=1}^n (x_k - \mu)^2$$

## Estimating $\mu$ and $\sigma^2$

### 1. Estimating $\mu$

To find the MLE for $\mu$, we take the derivative of $\ell(\mu, \sigma^2)$ with respect to $\mu$ and set it to zero:

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2} \sum_{k=1}^n (x_k - \mu) = 0$$

Solving for $\mu$:

$$\sum_{k=1}^n (x_k - \hat{\mu}) = 0 \implies \hat{\mu} = \frac{1}{n} \sum_{k=1}^n x_k$$

The MLE for $\mu$ is the **sample mean**.

### 2. Estimating $\sigma^2$

To find the MLE for $\sigma^2$, we take the derivative of $\ell(\mu, \sigma^2)$ with respect to $\sigma^2$ and set it to zero:

$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4} \sum_{k=1}^n (x_k - \mu)^2 = 0$$

Multiplying through by $2\sigma^4$:

$$-n\sigma^2 + \sum_{k=1}^n (x_k - \mu)^2 = 0$$

Solving for $\sigma^2$:

$$\hat{\sigma}^2 = \frac{1}{n} \sum_{k=1}^n (x_k - \hat{\mu})^2$$

The MLE for $\sigma^2$ is the **sample variance** (with denominator $n$ instead of $n-1$).

## Summary

The MLE estimates for the parameters of a univariate Gaussian distribution are:
- Mean: $\hat{\mu} = \frac{1}{n} \sum_{k=1}^n x_k$
- Variance: $\hat{\sigma}^2 = \frac{1}{n} \sum_{k=1}^n (x_k - \hat{\mu})^2$

These estimates maximize the likelihood of observing the given dataset under the assumed Gaussian model.