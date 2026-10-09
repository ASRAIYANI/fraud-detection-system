
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transactions.csv"

def main():
    # Load dataset
    df = pd.read_csv(DATA)
    y = df["is_fraud"].astype(int)
    X = df.drop(columns=["is_fraud", "transaction_id"])

    # Identify column types
    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = X.select_dtypes(exclude=["number"]).columns.tolist()

    # Preprocessing pipeline
    prep = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            num,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            cat,
        ),
    ])

    # Re-create the same 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    # Train on training data only
    model = Pipeline([
        ("preprocessor", prep),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    # Evaluate once on the untouched test set
    test_probabilities = model.predict_proba(X_test)[:, 1]

    # Threshold chosen using validation results and assumed costs
    threshold = 0.40
    predictions = (test_probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    pr_auc = average_precision_score(
        y_test, test_probabilities
    )
    roc_auc = roc_auc_score(
        y_test, test_probabilities
    )

    # Illustrative costs only
    estimated_cost = fp * 100 + fn * 5000

    print("\nFINAL TEST RESULTS")
    print("------------------")
    print(f"Threshold: {threshold:.2f}")
    print(f"Test transactions: {len(y_test)}")
    print(f"Actual fraud cases: {int(y_test.sum())}")
    print(f"Transactions flagged: {int(predictions.sum())}")
    print(f"True positives: {tp}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True negatives: {tn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Illustrative estimated cost: Rs. {estimated_cost:,}")


if __name__ == "__main__":
    main()