# Transaction Fraud Detection System

## 🚀 Live Demo

Try the deployed Fraud Detection System here:

**[Open the Live App](https://asraiyani-fraud-detection.streamlit.app/)

> Educational machine learning project using synthetic transaction data. Not intended for real-world banking or financial decisions.

## Skills
- Pandas data preparation and class imbalance
- scikit-learn preprocessing pipelines
- Logistic Regression and Random Forest
- Precision, recall, F1, ROC-AUC and PR-AUC / average precision
- Probability thresholds and false-positive/false-negative trade-offs
- joblib model persistence and Streamlit + Plotly

## Structure
```text
fraud_detection_system/
├── data/transactions.csv
├── src/train.py
├── app/app.py
├── models/                  # trained model generated here
├── reports/                 # evaluation reports generated here
├── requirements.txt
└── README.md
```

## Windows PowerShell setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/train.py
streamlit run app/app.py
```
If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```
Or run directly without activation:
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src/train.py
.venv\Scripts\python.exe -m streamlit run app/app.py
```

## Data
Target `is_fraud`: 1 = synthetic fraud example, 0 = legitimate. `transaction_id` is an identifier and is excluded from model training. Features include amount, hour, day of week, channel, merchant category, distance from home, recent transaction count, card-present flag, account age, international transaction and device mismatch.

## Method
1. Stratified 80/20 train-test split.
2. Numeric median imputation and scaling.
3. Categorical most-frequent imputation and one-hot encoding.
4. Compare class-weighted Logistic Regression and Random Forest.
5. Report PR-AUC (average precision), ROC-AUC, precision, recall and F1 at threshold 0.5.
6. Select the model with the higher average precision and save the fitted pipeline.

For rigorous model selection, tune on validation data or cross-validation and evaluate the final test set only once. This compact educational version compares models on the same holdout, so treat results as exploratory.

## Why accuracy is not enough
A model that labels every transaction legitimate can have high accuracy when fraud is rare while detecting no fraud. Precision measures how many flagged cases are actually fraud; recall measures how many fraud cases were found. F1 balances precision and recall. PR-AUC evaluates precision-recall performance across thresholds and is useful for rare positive classes.

## Interview summary
"I built a transaction fraud detection pipeline using Python, Pandas and scikit-learn. I handled missing values and categorical features using a preprocessing pipeline, compared class-weighted Logistic Regression with Random Forest, and evaluated precision, recall, F1, ROC-AUC and PR-AUC because fraud is an imbalanced classification problem. I saved the selected model pipeline with joblib and created a Streamlit dashboard to score example transactions and visualise model results. The data is synthetic, so the project demonstrates the workflow rather than real-world fraud performance."

## Improvements
- Time-based validation and leakage checks
- Tune the decision threshold to reflect fraud cost and analyst capacity
- Add probability calibration, explainability and drift monitoring
- Use authorised, privacy-protected real transaction data
- Require human review before taking action on flagged transactions
