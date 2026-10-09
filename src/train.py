
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transactions.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def main():
    # 1. Load the dataset
    df = pd.read_csv(DATA)

    y = df["is_fraud"].astype(int)
    X = df.drop(columns=["is_fraud", "transaction_id"])

    # 2. Identify numerical and categorical columns
    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = X.select_dtypes(exclude=["number"]).columns.tolist()

    # 3. Build preprocessing steps
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

    # 4. Split: 70% train, 15% validation, 15% test
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

    # 5. Define models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }

    # 6. Train and compare using validation data only
    rows = []
    fitted_models = {}

    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", prep),
            ("model", estimator),
        ])

        pipe.fit(X_train, y_train)
        val_scores = pipe.predict_proba(X_val)[:, 1]
        val_pred_05 = (val_scores >= 0.5).astype(int)

        val_ap = average_precision_score(y_val, val_scores)
        val_auc = roc_auc_score(y_val, val_scores)

        rows.append({
            "model": name,
            "validation_pr_auc": float(val_ap),
            "validation_roc_auc": float(val_auc),
            "precision_at_0_5": float(
                precision_score(y_val, val_pred_05, zero_division=0)
            ),
            "recall_at_0_5": float(
                recall_score(y_val, val_pred_05, zero_division=0)
            ),
            "f1_at_0_5": float(
                f1_score(y_val, val_pred_05, zero_division=0)
            ),
        })

        fitted_models[name] = pipe

    comparison = pd.DataFrame(rows).sort_values(
        "validation_pr_auc", ascending=False
    )
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)

    # 7. Select the model using validation PR-AUC
    best_name = comparison.iloc[0]["model"]
    best_pipe = fitted_models[best_name]

    # 8. Choose a threshold using validation data only
    val_scores = best_pipe.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_val, val_scores
    )

    # The final precision/recall entry has no corresponding threshold.
    p = precision[:-1]
    r = recall[:-1]
    f1_values = 2 * p * r / (p + r + 1e-12)

    best_index = int(f1_values.argmax())
    selected_threshold = float(thresholds[best_index])

    # 9. Evaluate the selected model ONCE on the final test set
    test_scores = best_pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_scores >= selected_threshold).astype(int)

    test_metrics = {
        "selected_model": best_name,
        "threshold_selected_on_validation": selected_threshold,
        "test_rows": int(len(X_test)),
        "fraud_cases_in_test": int(y_test.sum()),
        "test_pr_auc_average_precision": float(
            average_precision_score(y_test, test_scores)
        ),
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        "test_precision": float(
            precision_score(y_test, test_pred, zero_division=0)
        ),
        "test_recall": float(
            recall_score(y_test, test_pred, zero_division=0)
        ),
        "test_f1": float(
            f1_score(y_test, test_pred, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, test_pred
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            test_pred,
            output_dict=True,
            zero_division=0,
        ),
        "note": (
            "Synthetic educational data. Results are illustrative "
            "and do not establish real-world fraud performance."
        ),
    }

    # 10. Save the selected model and reports
    joblib.dump(best_pipe, MODELS / "fraud_pipeline.joblib")

    (REPORTS / "final_test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2)
    )

    # Save predictions for the held-out test transactions
    output = X_test.copy()
    output.insert(
        0, "transaction_id", df.loc[X_test.index, "transaction_id"]
    )
    output["actual_is_fraud"] = y_test
    output["fraud_probability"] = test_scores
    output["predicted_fraud"] = test_pred

    output.to_csv(REPORTS / "test_predictions.csv", index=False)

    summary = {
        "rows": int(len(df)),
        "fraud_count": int(y.sum()),
        "fraud_rate_percent": round(float(y.mean() * 100), 2),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "selected_model": best_name,
        "validation_pr_auc": float(
            comparison.iloc[0]["validation_pr_auc"]
        ),
        "threshold_selected_on_validation": selected_threshold,
    }

    (REPORTS / "training_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # 11. Display results
    print("\nMODEL COMPARISON (VALIDATION DATA)")
    print(comparison.to_string(index=False))

    print("\nFINAL TEST METRICS")
    print(json.dumps(test_metrics, indent=2))

    print("\nTRAINING SUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
    