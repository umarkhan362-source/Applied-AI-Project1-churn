# Customer Churn Prediction 
 
## Week 1: Exploratory Data Analysis 
 
### Dataset - Source: Telco Customer Churn (Kaggle) - Size: 7,043 customers, 21 features - Target: Predict customer churn (Yes/No) 
 
### Key Findings 
- **26.54% of customers churned**, while **73.46% remained**.
- Customers with **shorter tenure** generally have higher churn.
- **Higher monthly charges** are associated with increased churn.
- **Month-to-month contract customers** show higher churn compared with customers on longer-term contracts.
- **Internet service type** shows differences in churn rates.
- **Payment method** also shows noticeable differences in churn.
- `TotalCharges` generally increases with **customer tenure**. 
 
### Setup 
Open the Kaggle notebook or run locally: 
pip install pandas numpy matplotlib seaborn# Project_1

## Week 2: Building ML Models

* **Baseline (always "stay"):** accuracy **73.5%**
* **Best model:** **Logistic Regression**, AUC **0.842**, recall **56.7%** at threshold **0.50**
* **Top churn drivers (permutation importance):** **Tenure**, **TotalCharges**, **Contract_Two year**
* **Threshold chosen:** **0.15**, because missing a churner costs **PKR 6,000**, while an unnecessary retention offer costs **PKR 1,000**. The cost-based theoretical threshold is about **0.14**, and the empirical minimum-cost threshold from the notebook is **0.15**.
* **Engineered features:** **n_services, is_new, charge_per_mo, price_jump**; effect on AUC: **0.8422 → 0.8420**
* **Biggest lesson:** **Accuracy alone is not enough for churn prediction; recall, AUC, class imbalance, threshold selection, and business costs must also be considered.**

## Week 3: Model Optimization and Unsupervised Learning

**Dataset:** Telco Customer Churn (7,043 customers, 30 features after encoding) | **Split:** 5,634 train / 1,409 test 

### Key results at a glance

| Item | Result |
|---|---|
| Split-to-split accuracy range (20 seeds) | **0.780 to 0.828** (std 0.0104, theoretical SE 0.0107, 95% CI ± 0.021) |
| 5-fold CV AUC, Logistic Regression | 0.846 ± 0.013 |
| 5-fold CV AUC, Random Forest | 0.844 ± 0.011 |
| 5-fold CV AUC, XGBoost (tuned) | 0.850 ± 0.013 (0.8504 ± 0.0125) |
| Best Random Forest (random search) | max_depth 15, min_samples_leaf 15, max_features 0.213 |
| Grid vs random search | AUC 0.8468 vs 0.8464, 65 s vs 70 s, 120 fits each |
| Final model (XGBoost, test set used once) | **AUC 0.8478**, recall 0.529, precision 0.662 |
| Customer segments (k = 4) | churn 43%, 32%, 14%, 5% |
| PCA | 15 of 30 components explain 90% of the variance |

---

### 1. How noisy is one split?

The same Logistic Regression was scored on 20 different validation splits (n = 1,409 rows each).

Accuracy is a proportion, so its standard error is

$$SE = \sqrt{\frac{p(1-p)}{n}} = \sqrt{\frac{0.80 \times 0.20}{1409}} = 0.0107, \qquad 95\%\ \text{CI} = \pm 1.96 \times SE = \pm 0.021$$

| Quantity | Value |
|---|---|
| Minimum accuracy | 0.780 |
| Maximum accuracy | 0.828 |
| Range | 0.048 (≈ 4.5 SE) |
| Observed std over 20 seeds | 0.0104 |
| Theoretical SE | 0.0107 |
| Ratio observed / theoretical | 0.97 |

SE of the **difference** between two models scored on different splits: $\sqrt{2}\times 0.0107 = 0.0151$ (95% interval ± 2.9 points). Gaps between LR and RF (about 0.2 AUC points) are far smaller, so a single split cannot rank them.

### 2. 5-fold cross-validation

$$CV = \frac{1}{k}\sum_{i=1}^{k} s_i \qquad\qquad std = \sqrt{\frac{1}{k}\sum_{i=1}^{k}(s_i-CV)^2}$$

Illustration with five fold AUCs 0.826, 0.838, 0.846, 0.856, 0.864: mean = 4.230 / 5 = 0.846; squared deviations sum to 0.000888, so std = √(0.000888 / 5) = 0.0133.

| Model | AUC | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.846 ± 0.013 | 0.545 ± 0.042 | 0.594 ± 0.030 |
| Random Forest | 0.844 ± 0.011 | 0.496 ± 0.019 | 0.573 ± 0.020 |
| Gap (LR - RF) in std units | 0.002 / 0.012 = **0.17 std** | 0.049 / 0.033 = **1.5 std** | 0.021 / 0.026 = **0.8 std** |

AUC is a tie; recall shows LR catches more churners at the 0.5 threshold.

### 3. Hyperparameter tuning

**Validation curve (LR):** best C = 10 (λ = 1/C = 0.1). CV AUC is flat (≈ 0.846) from C ≈ 0.1 to 100, so any value there is equivalent.

**Search cost:**

$$\text{fits} = \prod(\text{values}) \times k \qquad\qquad P(\text{top }5\%) = 1 - 0.95^{\,n}$$

| | Grid search | Random search |
|---|---|---|
| Settings tried | 4 × 3 × 2 = 24 | 24 |
| Fits (× 5 folds) | 120 | 120 |
| Time | 65 s (0.54 s/fit) | 70 s (0.58 s/fit) |
| Best CV AUC | 0.8468 | 0.8464 |
| Best settings | depth 8, leaf 20, sqrt | depth 15, leaf 15, 0.213 |
| P(top 5% found) | n/a | 1 - 0.95²⁴ = 70.8% |

Six hyperparameters with 5 values each: 5⁶ × 5 = **78,125 fits ≈ 11.7 hours** by grid, versus 60 × 5 = **300 fits ≈ 3 minutes** by random search (P = 1 - 0.95⁶⁰ = 95.4%).

### 4. XGBoost

Each tree fits the residuals $y-p$, the negative gradient of log-loss:

$$F_m(x) = F_{m-1}(x) + \eta\, h_m(x), \qquad F_0 = \ln\frac{0.265}{0.735} = -1.02, \qquad w^* = -\frac{G}{H+\lambda}$$

Worked leaf (50 customers, 30 churners, p = 0.265): $G = 50(0.265) - 30 = -16.75$, $H = 50(0.265)(0.735) = 9.74$, so with λ = 1: $w^* = 16.75/10.74 = 1.56$. With η = 0.03 the step is 0.047, moving p from 0.265 to 0.274.

| Item | Value |
|---|---|
| Trees chosen by early stopping | 216 (of 2,000 allowed; 316 built, 84% saved) |
| Validation AUC (one split, SE ≈ 0.015) | 0.8534 |
| Tuned CV AUC (30 random settings, 150 fits) | 0.8504 |
| Best settings | depth 2, η = 0.034, 476 trees, subsample 0.6, λ = 1.97 |
| Lead over LR | 0.8504 - 0.8464 = 0.0040 = **0.31 std** |

### 5. Customer segments (K-means, k = 4)

WCSS objective: $\min \sum_{k}\sum_{x\in C_k}\lVert x-\mu_k\rVert^2$. Silhouette: $s=\dfrac{b-a}{\max(a,b)}$. Features were standardized with $z=(x-\bar x)/\sigma$ (TotalCharges is 99.97% of unscaled variance).

Hand check (points 1, 2, 3, 10, 11, 12; k = 2; start centres 1 and 2): WCSS 89.2 → 4.0 → 4.0 (converged), final centres 2.0 and 11.0.

| k | WCSS | Silhouette |
|---|---|---|
| 2 | ≈ 13,000 | ≈ 0.457 |
| 3 | ≈ 8,800 | ≈ 0.408 |
| **4** | ≈ 6,800 | ≈ 0.402 |
| 5 | ≈ 5,350 | ≈ 0.383 |

k = 4 chosen because silhouette barely changes from k = 3, the elbow still drops 23% at k = 4, and four groups are actionable.

| Segment | Customers | Tenure (mo) | Monthly | Services | Churn | Churners (size × rate) | Monthly revenue lost | Action |
|---|---|---|---|---|---|---|---|---|
| Mid-tenure, high spend | 2,157 | 18.4 | $80.41 | 3.28 | **43%** | ≈ 928 | ≈ $74,600 | Discounted 1-2 year contract offer |
| New, basic plan | 1,918 | 9.0 | $37.71 | 1.20 | 32% | ≈ 614 | ≈ $23,100 | Free add-on for 3 months |
| Loyal, premium | 1,938 | 59.8 | $92.09 | 5.06 | 14% | ≈ 271 | ≈ $25,000 | Loyalty reward, priority support |
| Loyal, low cost | 1,030 | 53.6 | $30.96 | 1.48 | 5% | ≈ 52 | ≈ $1,600 | Leave alone |

Check: 2,157 + 1,918 + 1,938 + 1,030 = 7,043; total churners ≈ 1,864, i.e. 26.5%.

### 6. PCA

$$\text{explained ratio}_j = \frac{\lambda_j}{\sum \lambda}, \qquad \text{total variance} = 30 \text{ (standardized)}, \qquad 90\% \times 30 = 27$$

15 of 30 components are needed (50% compression). Seven columns (`InternetService_No` and six "No internet service" dummies) are identical copies, so for the unit vector with weight $1/\sqrt{7}$ on them, $v^T\Sigma v = 7$. Hence $\lambda_1 \ge 7$ and PC1 explains at least $7/30 = 23.3\%$ (observed ≈ 33%). Their loadings are 0.302 each, so they hold $7 \times 0.302^2 = 63.8\%$ of PC1. Consequence: individual Logistic Regression coefficients for these dummies are not interpretable.

### 7. Final test (used once)

Chosen by highest CV AUC: **XGBoost (tuned)**. Reconstructed confusion matrix (374 churners, 1,035 stayers):

| | Predicted churn | Predicted stay |
|---|---|---|
| **Actual churn** | TP ≈ 198 | FN ≈ 176 |
| **Actual stay** | FP ≈ 101 | TN ≈ 934 |

$$\text{recall}=\frac{198}{374}=0.529,\quad \text{precision}=\frac{198}{299}=0.662,\quad F_1=\frac{2PR}{P+R}=0.588,\quad \text{accuracy}=\frac{198+934}{1409}=0.803$$

| Check | Value |
|---|---|
| Test AUC | 0.8478 (95% CI ≈ ± 0.026) |
| CV mean ± 2 std | 0.8504 ± 0.025 = [0.8254, 0.8754] → test AUC inside ✓ |
| Test minus CV mean | -0.0026 (-0.21 std) |
| Majority-class accuracy baseline | 0.735 |

### Biggest lesson

One split moved accuracy by 4.8 points, so XGBoost's 0.004 AUC edge over tuned logistic regression (0.3 std) is noise, not a win. Report every number as mean ± std and touch the test set once.

