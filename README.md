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
* **Best model:** **Logistic Regression**, AUC **0.842**, recall **56.7%** at threshold **0.5**
* **Top churn drivers (permutation importance):** **Tenure**, **TotalCharges**, **Contract_Two year**
* **Threshold chosen:** **0.15**, because **the business cost of missing a churner is higher than the cost of making an unnecessary retention offer**
* **Engineered features:** **n_services, is_new, charge_per_mo, price_jump**; effect on AUC: **0.8422 → 0.8420**
* **Biggest lesson:** **Model performance should be evaluated using more than accuracy; recall, AUC, threshold selection, class imbalance, and business costs are also important.**

