# Mathematical Foundations of Uncertainty Quantification

A rigorous mathematical treatment of uncertainty quantification concepts for air quality modeling.

**Based on:** Johns Hopkins UQ Course, Rasmussen & Williams (2006), and foundational UQ literature.

---

## Table of Contents

1. [Probability Foundations](#1-probability-foundations)
2. [Types of Uncertainty](#2-types-of-uncertainty)
3. [Random Variables and Distributions](#3-random-variables-and-distributions)
4. [Uncertainty Propagation](#4-uncertainty-propagation)
5. [Gaussian Processes](#5-gaussian-processes)
6. [Law of Total Variance](#6-law-of-total-variance)
7. [Bayesian Inference](#7-bayesian-inference)
8. [Conformal Prediction](#8-conformal-prediction)
9. [Calibration Metrics](#9-calibration-metrics)
10. [Application to Your System](#10-application-to-your-system)

---

## 1. Probability Foundations

### 1.1 Sample Space and Events

**Definition 1.1 (Sample Space):** The sample space Ω is the set of all possible outcomes of a random experiment.

**Definition 1.2 (Event):** An event A is a subset of the sample space, A ⊆ Ω.

**Definition 1.3 (σ-algebra):** A σ-algebra ℱ on Ω is a collection of subsets satisfying:
1. Ω ∈ ℱ
2. If A ∈ ℱ, then Aᶜ ∈ ℱ (closed under complement)
3. If A₁, A₂, ... ∈ ℱ, then ⋃ᵢAᵢ ∈ ℱ (closed under countable union)

### 1.2 Axioms of Probability

**Definition 1.4 (Probability Measure):** A probability measure P: ℱ → [0,1] satisfies:

1. **Non-negativity:** P(A) ≥ 0 for all A ∈ ℱ
2. **Normalization:** P(Ω) = 1
3. **Countable Additivity:** For disjoint events A₁, A₂, ...:
   ```
   P(⋃ᵢAᵢ) = ∑ᵢ P(Aᵢ)
   ```

**Theorem 1.1 (Properties of Probability):**
- P(∅) = 0
- P(Aᶜ) = 1 - P(A)
- If A ⊆ B, then P(A) ≤ P(B)
- P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

### 1.3 Conditional Probability

**Definition 1.5 (Conditional Probability):**
```
P(A|B) = P(A ∩ B) / P(B),  provided P(B) > 0
```

**Theorem 1.2 (Bayes' Theorem):**
```
P(A|B) = P(B|A)P(A) / P(B)
```

**Theorem 1.3 (Law of Total Probability):**
If {B₁, B₂, ..., Bₙ} is a partition of Ω:
```
P(A) = ∑ᵢ P(A|Bᵢ)P(Bᵢ)
```

---

## 2. Types of Uncertainty

### 2.1 Aleatory Uncertainty (Irreducible)

**Definition 2.1 (Aleatory Uncertainty):** Uncertainty arising from inherent randomness in the system. Cannot be reduced by gathering more data.

**Mathematical characterization:**
- Modeled as random variables with fixed distributions
- Represents measurement noise, natural variability
- In your system: sensor measurement error

**Example:** PM2.5 measurement noise
```
y = f(x) + ε,  where ε ~ N(0, σ²ₙ)
```
The noise variance σ²ₙ is aleatory - more measurements don't reduce it.

### 2.2 Epistemic Uncertainty (Reducible)

**Definition 2.2 (Epistemic Uncertainty):** Uncertainty arising from lack of knowledge. Can be reduced by gathering more data or improving the model.

**Mathematical characterization:**
- Modeled as uncertainty over model parameters or functions
- Represents data sparsity, model limitations
- In your system: GP posterior variance in data-sparse regions

**Example:** GP prediction in unobserved region
```
Var[f(x*)] = k(x*, x*) - k(x*, X)[K + σ²I]⁻¹k(X, x*)
```
This variance decreases as we add training points near x*.

### 2.3 Formal Distinction

**Theorem 2.1 (Uncertainty Decomposition):**
Total uncertainty can be decomposed as:
```
Var[Y] = E[Var[Y|θ]] + Var[E[Y|θ]]
         \_________/   \__________/
          Aleatory      Epistemic
```

Where θ represents unknown parameters or latent functions.

**In your air quality context:**
- **Aleatory:** Sensor noise, atmospheric turbulence, micro-scale variations
- **Epistemic:** Unknown pollution sources, sparse monitoring, model misspecification

---

## 3. Random Variables and Distributions

### 3.1 Random Variables

**Definition 3.1 (Random Variable):** A random variable X is a measurable function X: Ω → ℝ.

**Definition 3.2 (Cumulative Distribution Function):**
```
F_X(x) = P(X ≤ x)
```

**Definition 3.3 (Probability Density Function):** For continuous X:
```
f_X(x) = dF_X(x)/dx
```
Such that P(a ≤ X ≤ b) = ∫ₐᵇ f_X(x)dx

### 3.2 Moments

**Definition 3.4 (Expected Value):**
```
E[X] = ∫_{-∞}^{∞} x f_X(x) dx
```

**Definition 3.5 (Variance):**
```
Var[X] = E[(X - E[X])²] = E[X²] - (E[X])²
```

**Definition 3.6 (Covariance):**
```
Cov[X, Y] = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]
```

### 3.3 The Gaussian Distribution

**Definition 3.7 (Univariate Gaussian):**
```
X ~ N(μ, σ²)  ⟺  f_X(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
```

**Definition 3.8 (Multivariate Gaussian):**
```
X ~ N(μ, Σ)  ⟺  f_X(x) = (2π)^(-d/2)|Σ|^(-1/2) exp(-½(x-μ)ᵀΣ⁻¹(x-μ))
```

**Theorem 3.1 (Conditioning Gaussians):** ⭐ KEY FOR GPs
If:
```
[X₁]     [μ₁]   [Σ₁₁  Σ₁₂]
[X₂] ~ N([μ₂], [Σ₂₁  Σ₂₂])
```

Then the conditional distribution is:
```
X₁|X₂ ~ N(μ₁ + Σ₁₂Σ₂₂⁻¹(X₂ - μ₂), Σ₁₁ - Σ₁₂Σ₂₂⁻¹Σ₂₁)
```

**This is the foundation of GP regression!**

---

## 4. Uncertainty Propagation

### 4.1 Problem Statement

Given:
- Input random variable X with known distribution
- Deterministic model Y = g(X)

Find: Distribution of Y (or at least its moments)

### 4.2 Analytical Propagation (Simple Cases)

**Linear transformation:**
If Y = aX + b and X ~ N(μ, σ²):
```
Y ~ N(aμ + b, a²σ²)
```

**Sum of independent variables:**
If Y = X₁ + X₂ with X₁ ⊥ X₂:
```
E[Y] = E[X₁] + E[X₂]
Var[Y] = Var[X₁] + Var[X₂]
```

### 4.3 Taylor Series Approximation

For Y = g(X) with X having mean μ and variance σ²:

**First-order (Delta method):**
```
E[Y] ≈ g(μ)
Var[Y] ≈ (g'(μ))² σ²
```

**Second-order:**
```
E[Y] ≈ g(μ) + ½g''(μ)σ²
Var[Y] ≈ (g'(μ))²σ² + ½(g''(μ))²σ⁴
```

**Multivariate case:**
For Y = g(X) with X ∈ ℝᵈ:
```
E[Y] ≈ g(μ)
Var[Y] ≈ ∇g(μ)ᵀ Σ ∇g(μ)
```

### 4.4 Monte Carlo Propagation

**Algorithm:** Monte Carlo Uncertainty Propagation
```
1. Draw N samples: X⁽¹⁾, X⁽²⁾, ..., X⁽ᴺ⁾ ~ p(X)
2. Evaluate model: Y⁽ⁱ⁾ = g(X⁽ⁱ⁾) for i = 1, ..., N
3. Estimate moments:
   E[Y] ≈ (1/N) ∑ᵢ Y⁽ⁱ⁾
   Var[Y] ≈ (1/(N-1)) ∑ᵢ (Y⁽ⁱ⁾ - Ȳ)²
```

**Theorem 4.1 (Monte Carlo Convergence):**
The Monte Carlo estimator converges at rate O(1/√N), independent of dimension.

**In your system:** Bootstrap ensemble is Monte Carlo over hyperparameter uncertainty!

---

## 5. Gaussian Processes

### 5.1 Definition

**Definition 5.1 (Gaussian Process):**
A Gaussian Process is a collection of random variables, any finite number of which have a joint Gaussian distribution.

A GP is fully specified by:
- Mean function: m(x) = E[f(x)]
- Covariance function: k(x, x') = E[(f(x) - m(x))(f(x') - m(x'))]

We write: **f(x) ~ GP(m(x), k(x, x'))**

### 5.2 Covariance Functions (Kernels)

**Definition 5.2 (Valid Covariance Function):**
k(x, x') is a valid covariance function iff for any set of points {x₁, ..., xₙ}, the matrix K with Kᵢⱼ = k(xᵢ, xⱼ) is positive semi-definite.

**Common Kernels:**

**Squared Exponential (RBF):**
```
k(x, x') = σ² exp(-||x - x'||²/(2ℓ²))
```
- σ² = signal variance (output scale)
- ℓ = lengthscale (how far correlations extend)

**Matérn-ν:**
```
k(x, x') = σ² (2^(1-ν)/Γ(ν)) (√(2ν)r/ℓ)^ν K_ν(√(2ν)r/ℓ)
```
where r = ||x - x'|| and K_ν is modified Bessel function.

Special cases:
- ν = 1/2: Exponential (rough)
- ν = 3/2: Once differentiable
- ν = 5/2: Twice differentiable
- ν → ∞: Squared exponential (infinitely smooth)

### 5.3 GP Regression ⭐ KEY SECTION

**Problem:** Given training data D = {(xᵢ, yᵢ)}ᵢ₌₁ⁿ with yᵢ = f(xᵢ) + εᵢ, ε ~ N(0, σ²ₙ), predict f(x*) at new point x*.

**Prior:**
```
f ~ GP(0, k(x, x'))
```

**Joint distribution:**
```
[f ]     [K    k*]
[f*] ~ N([k*ᵀ  k**], 0)
```

Where:
- K = k(X, X) ∈ ℝⁿˣⁿ (training covariance)
- k* = k(X, x*) ∈ ℝⁿ (train-test covariance)
- k** = k(x*, x*) ∈ ℝ (test variance)

**Posterior (Equations 2.25-2.26 from R&W):**

Using Theorem 3.1 (conditioning Gaussians):

**Predictive Mean:**
```
μ* = E[f*|X, y, x*] = k*ᵀ(K + σ²ₙI)⁻¹y
```

**Predictive Variance:**
```
σ²* = Var[f*|X, y, x*] = k** - k*ᵀ(K + σ²ₙI)⁻¹k*
```

**Interpretation:**
- μ* = weighted combination of training outputs (weights depend on kernel similarity)
- σ²* = prior variance minus reduction from observing data
- σ²* does NOT depend on y, only on input locations!

### 5.4 Marginal Likelihood

**Definition 5.3 (Marginal Likelihood):**
```
p(y|X, θ) = ∫ p(y|f, X) p(f|X, θ) df
```

For GP regression with Gaussian noise:
```
y|X, θ ~ N(0, K + σ²ₙI)
```

**Log Marginal Likelihood:**
```
log p(y|X, θ) = -½ yᵀ(K + σ²ₙI)⁻¹y - ½ log|K + σ²ₙI| - (n/2)log(2π)
               \_________________/   \_______________/   \__________/
                  Data fit             Complexity          Constant
```

**Used for:** Hyperparameter optimization (maximize w.r.t. θ = {ℓ, σ², σ²ₙ})

---

## 6. Law of Total Variance

### 6.1 The Theorem

**Theorem 6.1 (Law of Total Variance):** ⭐ KEY FOR YOUR DECOMPOSITION
```
Var[Y] = E[Var[Y|X]] + Var[E[Y|X]]
```

**Proof:**
```
Var[Y] = E[Y²] - (E[Y])²

E[Y²] = E[E[Y²|X]]  (law of total expectation)

E[Y²|X] = Var[Y|X] + (E[Y|X])²  (definition of variance)

Therefore:
E[Y²] = E[Var[Y|X]] + E[(E[Y|X])²]

And:
(E[Y])² = (E[E[Y|X]])²

So:
Var[Y] = E[Var[Y|X]] + E[(E[Y|X])²] - (E[E[Y|X]])²
       = E[Var[Y|X]] + Var[E[Y|X]]  ∎
```

### 6.2 Application to Uncertainty Decomposition

**For your GP system:**

Let Y = f(x) + ε be the observation, with:
- f(x)|θ ~ GP posterior (depends on hyperparameters θ)
- ε ~ N(0, σ²ₙ) independent noise
- θ ~ p(θ|D) posterior over hyperparameters

**Applying Law of Total Variance:**
```
Var[Y] = E_θ[Var[Y|θ]] + Var_θ[E[Y|θ]]
         \____________/   \_____________/
         Within-model     Between-model
         uncertainty      uncertainty
```

**Further decomposition:**
```
Var[Y|θ] = Var[f(x)|θ] + σ²ₙ
           \__________/   \_/
            Epistemic    Aleatory
            (given θ)
```

**Your system computes:**
```python
total_var = ensemble_results.total_variance
within_var = ensemble_results.within_model_variance   # E[Var[Y|θ]]
between_var = ensemble_results.between_model_variance  # Var[E[Y|θ]]
```

### 6.3 Epistemic vs Aleatoric in GPs

**Epistemic uncertainty (model uncertainty):**
```
σ²_epistemic = k** - k*ᵀ(K + σ²ₙI)⁻¹k*
```
- Decreases as training points approach x*
- Reducible by collecting more data

**Aleatoric uncertainty (noise):**
```
σ²_aleatoric = σ²ₙ
```
- Constant (doesn't depend on x*)
- Irreducible by collecting more data at same locations

**Total predictive variance:**
```
σ²_total = σ²_epistemic + σ²_aleatoric
```

---

## 7. Bayesian Inference

### 7.1 Bayes' Theorem for Parameters

**Theorem 7.1 (Bayes' Theorem for Inference):**
```
p(θ|D) = p(D|θ)p(θ) / p(D)
         \____/ \__/   \__/
         Likelihood Prior  Evidence
```

Where:
- p(θ|D) = posterior (updated belief after seeing data)
- p(D|θ) = likelihood (probability of data given parameters)
- p(θ) = prior (initial belief)
- p(D) = evidence (normalizing constant)

### 7.2 Bayesian Linear Regression

**Model:**
```
y = Xw + ε,  ε ~ N(0, σ²I)
```

**Prior:** w ~ N(0, Σ_p)

**Posterior:**
```
p(w|X, y) = N(w|μ_w, Σ_w)
```
where:
```
Σ_w = (σ⁻²XᵀX + Σ_p⁻¹)⁻¹
μ_w = σ⁻²Σ_w Xᵀy
```

**Predictive distribution:**
```
p(y*|x*, X, y) = N(y*|x*ᵀμ_w, x*ᵀΣ_w x* + σ²)
```

### 7.3 Connection to GPs

**Theorem 7.2:** Bayesian linear regression with feature map φ(x) and prior w ~ N(0, Σ_p) is equivalent to GP regression with kernel:
```
k(x, x') = φ(x)ᵀ Σ_p φ(x')
```

**Implication:** GPs can be viewed as Bayesian linear regression with infinite-dimensional feature space!

### 7.4 Bayesian Model Averaging

**For hyperparameter uncertainty:**
```
p(y*|x*, D) = ∫ p(y*|x*, θ, D) p(θ|D) dθ
```

**Your bootstrap ensemble approximates this:**
```
p(y*|x*, D) ≈ (1/M) ∑ᵢ p(y*|x*, θ⁽ⁱ⁾, D)
```
where θ⁽ⁱ⁾ are bootstrap samples of hyperparameters.

---

## 8. Conformal Prediction

### 8.1 Problem Statement

**Goal:** Construct prediction intervals with guaranteed coverage, without distributional assumptions.

**Guarantee:** For any distribution P:
```
P(Y ∈ C(X)) ≥ 1 - α
```

### 8.2 Split Conformal Prediction

**Setup:**
- Training data: D_train
- Calibration data: D_cal = {(x₁, y₁), ..., (xₘ, yₘ)}
- Trained model: μ̂(x), σ̂(x)

**Algorithm:**
```
1. Compute nonconformity scores on calibration set:
   sᵢ = |yᵢ - μ̂(xᵢ)| / σ̂(xᵢ)  for i = 1, ..., m

2. Find quantile:
   q = ⌈(m+1)(1-α)⌉-th smallest value of {s₁, ..., sₘ}

3. Prediction interval for new x:
   C(x) = [μ̂(x) - q·σ̂(x), μ̂(x) + q·σ̂(x)]
```

**Theorem 8.1 (Coverage Guarantee):**
If (x₁, y₁), ..., (xₘ, yₘ), (x_{m+1}, y_{m+1}) are exchangeable:
```
P(y_{m+1} ∈ C(x_{m+1})) ≥ 1 - α
```

### 8.3 Your Implementation

```python
# In src/uncertainty/conformal.py
class ConformalPredictor:
    def calibrate(self, X_cal, y_cal):
        # Compute predictions
        mean, var = self.model.predict(X_cal)
        std = np.sqrt(var)

        # Compute nonconformity scores
        scores = np.abs(y_cal - mean) / std

        # Find quantile
        self.q = np.quantile(scores, 1 - self.alpha)

    def predict_interval(self, X_test):
        mean, var = self.model.predict(X_test)
        std = np.sqrt(var)

        lower = mean - self.q * std
        upper = mean + self.q * std
        return lower, upper
```

---

## 9. Calibration Metrics

### 9.1 Prediction Interval Coverage Probability (PICP)

**Definition 9.1:**
```
PICP = (1/n) ∑ᵢ 𝟙{yᵢ ∈ [Lᵢ, Uᵢ]}
```

**Interpretation:**
- PICP ≈ 1 - α for well-calibrated model
- PICP < 1 - α → overconfident (intervals too narrow)
- PICP > 1 - α → underconfident (intervals too wide)

### 9.2 Expected Calibration Error (ECE)

**Definition 9.2:**
```
ECE = ∑ⱼ (nⱼ/n) |acc(j) - conf(j)|
```

Where:
- Predictions binned by confidence
- acc(j) = accuracy in bin j
- conf(j) = average confidence in bin j

**For regression (your case):**
```
ECE = (1/B) ∑ᵦ |PICP_b - (1-α_b)|
```
Where b indexes confidence levels.

### 9.3 Continuous Ranked Probability Score (CRPS)

**Definition 9.3:**
```
CRPS(F, y) = ∫_{-∞}^{∞} (F(z) - 𝟙{y ≤ z})² dz
```

**For Gaussian predictive distribution N(μ, σ²):**
```
CRPS = σ[z(Φ(z) - 1) + 2φ(z) - 1/√π]
```
where z = (y - μ)/σ, Φ is CDF, φ is PDF.

**Interpretation:** Lower is better. Combines calibration and sharpness.

### 9.4 Mean Prediction Interval Width (MPIW)

**Definition 9.4:**
```
MPIW = (1/n) ∑ᵢ (Uᵢ - Lᵢ)
```

**Goal:** Minimize MPIW while maintaining PICP ≈ 1 - α (sharpness vs calibration trade-off)

---

## 10. Application to Your System

### 10.1 Complete Mathematical Framework

Your FusionGP UQ system implements:

**1. GP Posterior (Section 5.3):**
```python
mean, var = model.predict_f(X)  # Equations 5.7, 5.8
```

**2. Uncertainty Decomposition (Section 6):**
```python
# Law of Total Variance
total_var = E[Var[Y|θ]] + Var[E[Y|θ]]
          = within_model_var + between_model_var
```

**3. Epistemic/Aleatoric Split (Section 6.3):**
```python
epistemic_var = σ²_posterior - σ²_noise
aleatoric_var = σ²_noise
```

**4. Bootstrap Ensemble (Section 4.4):**
```python
# Monte Carlo over hyperparameters
θ⁽¹⁾, θ⁽²⁾, ..., θ⁽ᴹ⁾ ~ bootstrap samples
E[Y] ≈ (1/M) ∑ᵢ μ(x; θ⁽ⁱ⁾)
Var[Y] ≈ (1/M) ∑ᵢ [σ²(x; θ⁽ⁱ⁾) + μ(x; θ⁽ⁱ⁾)²] - (E[Y])²
```

**5. Conformal Prediction (Section 8):**
```python
# Distribution-free coverage guarantee
C(x) = [μ̂(x) - q·σ̂(x), μ̂(x) + q·σ̂(x)]
P(Y ∈ C(X)) ≥ 0.95
```

**6. Calibration Evaluation (Section 9):**
```python
PICP = fraction of y in [lower, upper]  # Should be ≈ 0.95
ECE = calibration error across confidence levels
CRPS = combined calibration + sharpness score
```

### 10.2 Key Equations Summary

| Concept | Equation | Your Code |
|---------|----------|-----------|
| GP Mean | μ* = k*ᵀ(K + σ²I)⁻¹y | `model.predict_f(X)[0]` |
| GP Variance | σ²* = k** - k*ᵀ(K + σ²I)⁻¹k* | `model.predict_f(X)[1]` |
| Total Variance | Var[Y] = E[Var[Y\|θ]] + Var[E[Y\|θ]] | `ensemble.total_variance` |
| Epistemic | σ²_e = σ²_posterior - σ²_noise | `pred.epistemic_std**2` |
| Aleatory | σ²_a = σ²_noise | `pred.aleatoric_std**2` |
| Conformal | C(x) = μ̂(x) ± q·σ̂(x) | `pred.lower_95, pred.upper_95` |
| PICP | (1/n)∑𝟙{yᵢ ∈ Cᵢ} | `metrics['picp']` |

### 10.3 Mathematical Justification for Your Research Questions

**RQ1 (Decomposition):**
Justified by Law of Total Variance (Theorem 6.1). Your decomposition:
```
σ²_total = σ²_epistemic + σ²_aleatoric
```
is mathematically rigorous when:
- σ²_epistemic = GP posterior variance
- σ²_aleatoric = estimated noise variance

**RQ2 (Hyperparameter Uncertainty):**
Justified by Bayesian model averaging (Section 7.4). Point estimates underestimate because:
```
Var[Y] = E_θ[Var[Y|θ]] + Var_θ[E[Y|θ]]
                          \____________/
                          Missing in point estimate!
```

**RQ3 (Calibration):**
Justified by conformal prediction theory (Theorem 8.1). Your 95% intervals have:
```
P(Y ∈ C(X)) ≥ 0.95
```
regardless of true distribution.

**RQ4 (OOD Detection):**
Justified by GP variance behavior. For x* far from training data:
```
σ²* = k** - k*ᵀ(K + σ²I)⁻¹k* → k**
```
As k* → 0 (no correlation with training), variance approaches prior variance.

---

## References

1. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

2. Der Kiureghian, A., & Ditlevsen, O. (2009). Aleatory or epistemic? Does it matter? *Structural Safety*, 31(2), 105-112.

3. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.

4. Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359-378.

5. Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods*. Springer.

6. Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.

---

## Quick Reference Card

### Key Formulas

**GP Posterior:**
```
μ* = k*ᵀ(K + σ²I)⁻¹y
σ²* = k** - k*ᵀ(K + σ²I)⁻¹k*
```

**Law of Total Variance:**
```
Var[Y] = E[Var[Y|X]] + Var[E[Y|X]]
```

**RBF Kernel:**
```
k(x, x') = σ² exp(-||x - x'||²/(2ℓ²))
```

**Conformal Interval:**
```
C(x) = [μ̂(x) - q·σ̂(x), μ̂(x) + q·σ̂(x)]
```

**PICP:**
```
PICP = (1/n) ∑ᵢ 𝟙{yᵢ ∈ [Lᵢ, Uᵢ]}
```

---

**This document provides the mathematical rigor needed for your dissertation methods section.**
