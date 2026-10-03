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
- Split-to-split accuracy range across 20 seeds: 0.780 to 0.828 (std 0.0104, theoretical SE 0.0107, 95% CI +/- 0.021)
- 5-fold CV AUC: LR 0.846 +/- 0.013, RF 0.844 +/- 0.011, XGBoost 0.850 +/- 0.013 (tuned; 0.8504 +/- 0.0125)
- Tuning: best RF params {max_depth 15, min_samples_leaf 15, max_features 0.213} (random search; grid best: depth 8, leaf 20, sqrt); grid vs random search time 65 s vs 70 s (120 fits each, AUC 0.8468 vs 0.8464)
- Test AUC of final model (XGBoost, used once): 0.8478 (recall 0.529, precision 0.662)
- Customer segments (k = 4): [Mid-tenure high spend, churn 43%], [New basic plan, churn 32%], [Loyal premium, churn 14%], [Loyal low cost, churn 5%]
- PCA: 15 of 30 components explain 90% of the variance
- Biggest lesson: one split moved accuracy by 4.8 points, so XGBoost's 0.004 AUC edge over tuned logistic regression (0.3 std) is noise, not a win.

