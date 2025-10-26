## EDA and DL Model

# Credit Risk Modeling — EDA, Deep Learning (ANN) & Offline RL

Project: end-to-end exploration, feature engineering, supervised deep model (ANN) and an offline RL agent for loan approval using the LendingClub dataset (accepted_2007_to_2018Q4.csv).

## Repository contents
- credit-risk-modeling-deep-learning-ann.ipynb — Full EDA, preprocessing, feature selection, supervised models (Logistic Regression, Random Forest, Gradient Boosting, MLP) and model saving. Key variables: `selected_features_names`, `df_final`, saved artifacts `nn_model_model.pkl`, `rf_soft_model.pkl`, `lr_model.pkl`, `df_2014-18_selected.csv`.
- offline_rl_lendingclub.ipynb — Offline RL pipeline: dataset → MDP dataset → offline RL training and evaluation (reward engineering included).
- utils.py — plotting and helper utilities used by notebooks.
- (optional) saved model files and CSV produced by notebooks.

## Summary of workflow
1. EDA and cleaning
   - Load a small sample then full CSV (`accepted_2007_to_2018Q4.csv`).
   - Map `loan_status` to binary `loan_status_binary` (good / bad).
   - Drop columns with >50% missing; parse dates; normalize percent fields (e.g., `int_rate%`); map flags (e.g., `debt_settlement_flag`).
   - Extract date-derived features (loan_age, time_since_last_payment, etc.).
   - Use WoE/IV for categorical assessment (function: `process_categorical_columns`).

2. Feature engineering & selection
   - One-hot encode categorical variables, impute missing numeric values (mean), scale when needed.
   - Remove highly correlated / low-IV features; create derived features (e.g., `loan_amnt_div_instlmnt`).
   - Use RandomForest + SelectFromModel to pick `selected_features_names`.

3. Supervised DL model (ANN)
   - Train a Keras/TensorFlow ANN and scikit-learn MLP (notebook includes both variants).
   - Evaluate with AUC and F1 (and F2, recall, accuracy). Evaluation helper: `evaluate_model`.

4. Offline RL formulation
   - State: preprocessed feature vector for an applicant.
   - Action: {0: Deny, 1: Approve}.
   - Reward: deny → 0; approve & paid → + (loan_amnt * int_rate); approve & default → -loan_amnt.
   - Train an offline RL agent (recommended libraries: d3rlpy, d4rl). Report estimated policy value vs baselines.

## How to reproduce (minimal steps)
1. Create and activate a virtual environment (Windows):


