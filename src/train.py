"""
ML Training Pipeline: Loads validated Gold data, performs leak-free preprocessing,
trains a delivery delay classifier, evaluates performance, and persists artifacts.
Supports modular execution for Airflow tasks and unified execution via CLI.
"""

import argparse
import glob
import json
import os
import shutil

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = [
    "order_hour",
    "distance_km",
    "items_count",
    "package_weight_kg",
    "order_value_egp",
    "promised_delivery_days",
    "warehouse_load_pct",
    "carrier_on_time_rate",
    "distance_per_promised_day",
    "weight_per_item",
    "is_rush_order",
]

CATEGORICAL_FEATURES = [
    "customer_region",
    "warehouse_id",
    "shipping_method",
    "weather_risk_level",
]


def load_gold_dataset(gold_dir: str = "data/gold") -> pd.DataFrame:
    """Load all available validated daily Gold CSV files."""
    gold_files = sorted(glob.glob(os.path.join(gold_dir, "orders_*.csv")))
    if not gold_files:
        raise FileNotFoundError(
            f"No Gold CSV files found in {gold_dir}. Run ETL pipeline first."
        )

    dfs = [pd.read_csv(f) for f in gold_files]
    dataset = pd.concat(dfs, ignore_index=True)
    print(
        f"[Data Loader] Loaded {len(gold_files)} Gold files ({len(dataset)} total records)."
    )
    return dataset


# --- Modular steps for Airflow orchestration ---


def step_load_gold_data(gold_dir: str = "data/gold", ml_dir: str = "data/ml") -> str:
    """Step 1: Load gold data, split into train and test sets, and stage in data/ml."""
    os.makedirs(ml_dir, exist_ok=True)
    df = load_gold_dataset(gold_dir)

    target_col = "late_delivery"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in Gold dataset.")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[target_col]
    )

    train_path = os.path.join(ml_dir, "train.csv")
    test_path = os.path.join(ml_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(
        f"[Airflow: load_gold_data] Staged train ({len(train_df)} rows) and test ({len(test_df)} rows) in {ml_dir}"
    )
    return train_path


def step_train_model(ml_dir: str = "data/ml") -> str:
    """Step 2: Train leak-free Scikit-learn Pipeline on staged training data."""
    train_path = os.path.join(ml_dir, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Staged training data not found: {train_path}")

    train_df = pd.read_csv(train_path)
    target_col = "late_delivery"
    drop_cols = ["order_id", "order_date", target_col]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)

    num_cols = [c for c in NUMERIC_FEATURES if c in X_train.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    print(f"[Airflow: train_model] Fitting model on {len(X_train)} training records...")
    model_pipeline.fit(X_train, y_train)

    temp_model_path = os.path.join(ml_dir, "temp_model.joblib")
    joblib.dump(model_pipeline, temp_model_path)
    print(f"[Airflow: train_model] Staged fitted model -> {temp_model_path}")
    return temp_model_path


def step_evaluate_model(
    ml_dir: str = "data/ml", metrics_path: str = "metrics/metrics.json"
) -> dict:
    """Step 3: Evaluate fitted model on staged test data and export metrics JSON."""
    temp_model_path = os.path.join(ml_dir, "temp_model.joblib")
    test_path = os.path.join(ml_dir, "test.csv")
    train_path = os.path.join(ml_dir, "train.csv")

    model_pipeline = joblib.load(temp_model_path)
    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)

    target_col = "late_delivery"
    drop_cols = ["order_id", "order_date", target_col]
    feature_cols = [c for c in test_df.columns if c not in drop_cols]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int)

    y_pred = model_pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    dates = sorted(
        list(
            set(
                train_df["order_date"].unique().tolist()
                + test_df["order_date"].unique().tolist()
            )
        )
    )
    date_min = dates[0] if dates else "unknown"
    date_max = dates[-1] if dates else "unknown"

    metrics_payload = {
        "model_name": "RandomForestClassifier",
        "split_strategy": "stratified_split_80_20",
        "total_records": len(train_df) + len(test_df),
        "train_records": len(train_df),
        "test_records": len(test_df),
        "date_range": [date_min, date_max],
        "features": {
            "numeric": [c for c in NUMERIC_FEATURES if c in feature_cols],
            "categorical": [c for c in CATEGORICAL_FEATURES if c in feature_cols],
        },
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        },
    }

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(
        f"[Airflow: evaluate_model] Metrics: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}"
    )
    print(f"[Airflow: evaluate_model] Saved metrics -> {metrics_path}")
    return metrics_payload


def step_save_model(
    ml_dir: str = "data/ml",
    final_model_path: str = "models/delivery_delay_model.joblib",
) -> str:
    """Step 4: Promote and persist final model to models directory."""
    temp_model_path = os.path.join(ml_dir, "temp_model.joblib")
    if not os.path.exists(temp_model_path):
        raise FileNotFoundError(f"Model artifact not found: {temp_model_path}")

    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    shutil.copy2(temp_model_path, final_model_path)
    print(f"[Airflow: save_model] Successfully promoted model -> {final_model_path}")
    return final_model_path


def train_and_evaluate(
    gold_dir: str = "data/gold",
    ml_dir: str = "data/ml",
    model_path: str = "models/delivery_delay_model.joblib",
    metrics_path: str = "metrics/metrics.json",
) -> dict:
    """Unified full run (calls all 4 steps sequentially)."""
    step_load_gold_data(gold_dir=gold_dir, ml_dir=ml_dir)
    step_train_model(ml_dir=ml_dir)
    metrics = step_evaluate_model(ml_dir=ml_dir, metrics_path=metrics_path)
    step_save_model(ml_dir=ml_dir, final_model_path=model_path)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Late Delivery Classifier")
    parser.add_argument("--gold-dir", type=str, default="data/gold")
    parser.add_argument("--ml-dir", type=str, default="data/ml")
    parser.add_argument(
        "--model-path", type=str, default="models/delivery_delay_model.joblib"
    )
    parser.add_argument("--metrics-path", type=str, default="metrics/metrics.json")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["load", "train", "evaluate", "save", "all"],
        help="Run specific training step or all",
    )
    args = parser.parse_args()

    if args.step == "load":
        step_load_gold_data(args.gold_dir, args.ml_dir)
    elif args.step == "train":
        step_train_model(args.ml_dir)
    elif args.step == "evaluate":
        step_evaluate_model(args.ml_dir, args.metrics_path)
    elif args.step == "save":
        step_save_model(args.ml_dir, args.model_path)
    else:
        train_and_evaluate(
            gold_dir=args.gold_dir,
            ml_dir=args.ml_dir,
            model_path=args.model_path,
            metrics_path=args.metrics_path,
        )
