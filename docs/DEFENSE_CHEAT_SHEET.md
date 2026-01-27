# Defense Cheat Sheet: Key Equations You MUST Know

**Print this. Memorize it. They WILL ask about these.**

---

## 1. What is a Gaussian Process?

> "A GP is a collection of random variables, any finite number of which have a joint Gaussian distribution."

**Notation:**
```
f(x) ~ GP(m(x), k(x, x'))
```
- m(x) = mean function
- k(x, x') = covariance function (kernel)

**Why it matters:** GPs give you predictions WITH uncertainty automatically.

---

## 2. GP Posterior (THE CORE EQUATIONS)

Given training data (X, y) with noise y = f(x) + ε, ε ~ N(0, σ²ₙ):

### Predictive Mean (Equation 2.25):
```
μ* = k*ᵀ (K + σ²ₙI)⁻¹ y
```

### Predictive Variance (Equation 2.26):
```
σ²* = k** - k*ᵀ (K + σ²ₙI)⁻¹ k*
```

**Where:**
- K = k(X, X) = training covariance matrix (n × n)
- k* = k(x*, X) = test-train covariance (n × 1)
- k** = k(x*, x*) = test point prior variance (scalar)

**Key insight:** Variance does NOT depend on y, only on locations!

---

## 3. Law of Total Variance (YOUR DECOMPOSITION)

```
Var[Y] = E[Var[Y|θ]] + Var[E[Y|θ]]
         \_________/   \__________/
          Within        Between
          (Aleatory)    (Epistemic)
```

**In words:** Total uncertainty = Average uncertainty + Uncertainty in the average

**Your application:**
```
Total = Within-model variance + Between-model variance
      = E_θ[σ²(x;θ)]        + Var_θ[μ(x;θ)]
```

---

## 4. Epistemic vs Aleatoric

### Aleatory (Irreducible):
- **Definition:** Inherent randomness
- **Source:** Measurement noise, natural variability
- **Math:** σ²_aleatory = σ²_noise
- **Cannot be reduced** by collecting more data

### Epistemic (Reducible):
- **Definition:** Lack of knowledge
- **Source:** Data sparsity, model limitation
- **Math:** σ²_epistemic = σ²_posterior - σ²_noise
- **CAN be reduced** by collecting more data

**Your result:** ~60-80% epistemic far from sensors, ~60-70% aleatoric near sensors

---

## 5. RBF Kernel (Squared Exponential)

```
k(x, x') = σ² exp(-||x - x'||² / (2ℓ²))
```

**Parameters:**
- σ² = signal variance (vertical scale)
- ℓ = lengthscale (horizontal correlation distance)

**Lengthscale interpretation:**
- Small ℓ → wiggly function, needs many points
- Large ℓ → smooth function, can interpolate far
- Points beyond ~2.5ℓ from training → extrapolating (OOD)

---

## 6. Conformal Prediction

**Algorithm:**
1. Compute scores on calibration data: sᵢ = |yᵢ - μ̂ᵢ| / σ̂ᵢ
2. Find quantile: q = (1-α) quantile of scores
3. Interval: C(x) = [μ̂(x) - q·σ̂(x), μ̂(x) + q·σ̂(x)]

**Guarantee:**
```
P(Y ∈ C(X)) ≥ 1 - α
```

**Why it matters:** Distribution-free coverage guarantee!

---

## 7. Calibration Metrics

### PICP (Prediction Interval Coverage Probability):
```
PICP = (1/n) ∑ᵢ 𝟙{yᵢ ∈ [Lᵢ, Uᵢ]}
```
- Target: PICP ≈ 0.95 for 95% intervals
- Your result: PICP = 0.98 ✓

### ECE (Expected Calibration Error):
```
ECE = average |observed coverage - expected coverage|
```
- Target: ECE < 0.05
- Lower is better

### CRPS (Continuous Ranked Probability Score):
- Combines calibration + sharpness
- Lower is better

---

## 8. Bayes' Theorem

```
p(θ|D) = p(D|θ) × p(θ) / p(D)
         \____/   \__/   \__/
       Likelihood Prior Evidence

Posterior ∝ Likelihood × Prior
```

**GP connection:** GP posterior IS Bayesian inference over functions!

---

## 9. Bootstrap Ensemble (Hyperparameter UQ)

**Problem:** Standard GP uses point estimate of hyperparameters θ

**Solution:** Bootstrap ensemble
```
For i = 1 to M:
    D⁽ⁱ⁾ = bootstrap sample from D
    θ⁽ⁱ⁾ = optimize hyperparameters on D⁽ⁱ⁾

E[Y] = (1/M) ∑ᵢ μ(x; θ⁽ⁱ⁾)
Var[Y] = (1/M) ∑ᵢ [σ²(x; θ⁽ⁱ⁾) + μ²(x; θ⁽ⁱ⁾)] - E[Y]²
```

**Your finding:** Point estimates underestimate uncertainty by 10-30%

---

## 10. OOD Detection

**Principle:** GP variance increases far from training data

**Math:** As distance to training data increases:
```
k* → 0  (no correlation)
σ²* → k** (prior variance)
```

**Your threshold:** > 2.5 lengthscales from training → flag as OOD

**Your finding:** OOD detection improves coverage 87% → 95%

---

## Quick Defense Answers

### "What is your main contribution?"
> "A rigorous 7-layer uncertainty quantification framework for GP-based air quality fusion that decomposes uncertainty into epistemic and aleatoric components, quantifies hyperparameter uncertainty, provides distribution-free coverage guarantees, and translates uncertainty to policy-relevant outputs."

### "Why Gaussian Processes?"
> "GPs naturally provide uncertainty estimates through the posterior variance. Unlike neural networks which give point predictions, GPs give us both the mean prediction AND a principled uncertainty measure derived from Bayesian inference."

### "How do you decompose uncertainty?"
> "Using the Law of Total Variance: Total variance equals expected variance (aleatory) plus variance of expectation (epistemic). For GPs, the aleatory component is the noise variance, while the epistemic is the posterior variance that decreases near training data."

### "How do you know your uncertainties are correct?"
> "We validate through conformal prediction which provides distribution-free coverage guarantees, and calibration metrics including PICP (achieved 0.98 vs target 0.95) and ECE (achieved < 0.05)."

### "What's the practical impact?"
> "We can tell decision-makers not just 'PM2.5 is 35 μg/m³' but 'PM2.5 is 35±5 μg/m³ with 95% confidence, and 67% of this uncertainty can be reduced by deploying sensors at these specific locations.'"

---

## The 5 Equations to Write from Memory

1. **GP Mean:** μ* = k*ᵀ(K + σ²I)⁻¹y

2. **GP Variance:** σ²* = k** - k*ᵀ(K + σ²I)⁻¹k*

3. **Law of Total Variance:** Var[Y] = E[Var[Y|θ]] + Var[E[Y|θ]]

4. **RBF Kernel:** k(x,x') = σ²exp(-||x-x'||²/(2ℓ²))

5. **Bayes:** p(θ|D) ∝ p(D|θ)p(θ)

---

## Numbers to Remember

| Metric | Your Result | Target |
|--------|-------------|--------|
| PICP (95% CI) | 0.98 | ≈ 0.95 |
| ECE | < 0.05 | < 0.05 |
| Epistemic fraction (sparse) | 60-80% | - |
| Epistemic fraction (dense) | 30-40% | - |
| Hyperparameter underestimation | 10-30% | - |
| OOD coverage improvement | 87%→95% | - |

---

**YOU'VE GOT THIS! 💪**
