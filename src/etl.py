"""
Medallion ETL Pipeline: Bronze -> Silver -> Gold + Validation
Processes daily orders CSV for a specific date (date-aware, idempotent).
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd


def extract_to_bronze(
    date_str: str, raw_dir: str = "data/raw", bronze_dir: str = "data/bronze"
) -> str:
    """
    Bronze stage: Copy raw daily CSV into bronze storage without modifying any content.
    Preserves duplicates and invalid records to maintain a raw data audit trail.
    """
    os.makedirs(bronze_dir, exist_ok=True)
    raw_file = os.path.join(raw_dir, f"orders_{date_str}.csv")
    bronze_file = os.path.join(bronze_dir, f"orders_{date_str}.csv")

    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Source file not found: {raw_file}")

    # Overwrite if exists to guarantee idempotency
    shutil.copy2(raw_file, bronze_file)
    print(f"[Bronze] Successfully copied {raw_file} -> {bronze_file}")
    return bronze_file


def _clean_shipping_method(val: any) -> str:
    """Standardize shipping method string values."""
    if pd.isna(val) or str(val).strip() == "":
        return "Standard"
    cleaned = str(val).strip().lower().replace(" ", "_")
    if "same" in cleaned:
        return "Same_Day"
    elif "express" in cleaned:
        return "Express"
    elif "standard" in cleaned:
        return "Standard"
    return str(val).strip().title()


def _clean_weather_risk(val: any) -> str:
    """Standardize weather risk level strings."""
    if pd.isna(val) or str(val).strip() == "":
        return "Unknown"
    cleaned = str(val).strip().capitalize()
    return cleaned if cleaned in ["Low", "Medium", "High"] else "Unknown"


def transform_to_silver(
    date_str: str, bronze_dir: str = "data/bronze", silver_dir: str = "data/silver"
) -> str:
    """
    Silver stage: Deduplicate on order_id, standardize categorical strings,
    sanitize domain-invalid values (negative distance/price, 0 items), and enforce types.
    """
    os.makedirs(silver_dir, exist_ok=True)
    bronze_file = os.path.join(bronze_dir, f"orders_{date_str}.csv")
    silver_file = os.path.join(silver_dir, f"orders_{date_str}.csv")

    if not os.path.exists(bronze_file):
        raise FileNotFoundError(f"Bronze file not found: {bronze_file}")

    df = pd.read_csv(bronze_file)
    initial_count = len(df)

    # 1. Deduplicate by order_id
    df = df.drop_duplicates(subset=["order_id"], keep="first").copy()
    dedup_count = len(df)
    dropped_dups = initial_count - dedup_count

    # 2. String standardization
    df["order_id"] = df["order_id"].astype(str).str.strip()
    df["order_date"] = df["order_date"].astype(str).str.strip()
    df["customer_region"] = df["customer_region"].astype(str).str.strip()
    df["warehouse_id"] = df["warehouse_id"].astype(str).str.strip()
    df["shipping_method"] = df["shipping_method"].apply(_clean_shipping_method)
    df["weather_risk_level"] = df["weather_risk_level"].apply(_clean_weather_risk)

    # 3. Numeric conversion and domain validation
    # Numeric conversion
    num_cols = [
        "distance_km",
        "package_weight_kg",
        "order_value_egp",
        "warehouse_load_pct",
        "carrier_on_time_rate",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    int_cols = ["order_hour", "items_count", "promised_delivery_days", "late_delivery"]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Domain cleaning rules:
    # Negative distances (e.g., -8.0) are corrupt sensor/input values -> set to NaN for imputation
    df.loc[df["distance_km"] < 0, "distance_km"] = np.nan

    # Negative order values (e.g., -110.0) are invalid -> set to NaN
    df.loc[df["order_value_egp"] < 0, "order_value_egp"] = np.nan

    # items_count must be at least 1 (clip zero/negative records)
    df["items_count"] = df["items_count"].fillna(1).clip(lower=1).astype(int)

    # promised_delivery_days must be at least 1
    df["promised_delivery_days"] = (
        df["promised_delivery_days"].fillna(1).clip(lower=1).astype(int)
    )

    # carrier_on_time_rate must be in [0, 1]
    df.loc[
        (df["carrier_on_time_rate"] < 0) | (df["carrier_on_time_rate"] > 1),
        "carrier_on_time_rate",
    ] = np.nan

    # Ensure integer fields
    df["order_hour"] = df["order_hour"].fillna(0).astype(int)
    df["late_delivery"] = df["late_delivery"].astype(int)

    df.to_csv(silver_file, index=False)
    print(
        f"[Silver] Cleaned {date_str}: {initial_count} rows -> {dedup_count} rows ({dropped_dups} duplicates dropped). Saved to {silver_file}"
    )
    return silver_file


def prepare_gold(
    date_str: str, silver_dir: str = "data/silver", gold_dir: str = "data/gold"
) -> str:
    """
    Gold stage: Engineer business features from order attributes without target leakage.
    Preserves late_delivery target column.
    """
    os.makedirs(gold_dir, exist_ok=True)
    silver_file = os.path.join(silver_dir, f"orders_{date_str}.csv")
    gold_file = os.path.join(gold_dir, f"orders_{date_str}.csv")

    if not os.path.exists(silver_file):
        raise FileNotFoundError(f"Silver file not found: {silver_file}")

    df = pd.read_csv(silver_file)

    # Feature 1: distance_per_promised_day (Transit urgency: higher km/day increases delay risk)
    df["distance_per_promised_day"] = df["distance_km"] / df["promised_delivery_days"]

    # Feature 2: weight_per_item (Package density / handling complexity)
    df["weight_per_item"] = df["package_weight_kg"] / df["items_count"]

    # Feature 3: is_rush_order (Binary indicator for 1-day promise or Same-Day shipping)
    df["is_rush_order"] = (
        (df["promised_delivery_days"] == 1) | (df["shipping_method"] == "Same_Day")
    ).astype(int)

    df.to_csv(gold_file, index=False)
    print(
        f"[Gold] Prepared {date_str} with {len(df)} rows and 3 engineered features. Saved to {gold_file}"
    )
    return gold_file


def validate_gold(date_str: str, gold_dir: str = "data/gold") -> bool:
    """
    Validation stage: Asserts data quality criteria:
    - Non-empty dataset
    - All required columns present
    - No duplicate order_id
    - Correct partition date
    - Valid binary target labels (0 or 1)
    """
    gold_file = os.path.join(gold_dir, f"orders_{date_str}.csv")
    if not os.path.exists(gold_file):
        raise FileNotFoundError(f"Gold file not found for validation: {gold_file}")

    df = pd.read_csv(gold_file)

    # 1. Check non-empty
    if len(df) == 0:
        raise ValueError(
            f"[Validation Failed] Gold dataset for {date_str} is completely empty!"
        )

    # 2. Check required columns
    required_cols = [
        "order_id",
        "order_date",
        "order_hour",
        "customer_region",
        "warehouse_id",
        "shipping_method",
        "distance_km",
        "items_count",
        "package_weight_kg",
        "order_value_egp",
        "promised_delivery_days",
        "warehouse_load_pct",
        "weather_risk_level",
        "carrier_on_time_rate",
        "distance_per_promised_day",
        "weight_per_item",
        "is_rush_order",
        "late_delivery",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[Validation Failed] Missing required columns: {missing_cols}"
        )

    # 3. Check for duplicates
    dup_ids = df[df["order_id"].duplicated()]["order_id"].tolist()
    if dup_ids:
        raise ValueError(
            f"[Validation Failed] Found {len(dup_ids)} duplicate order_ids! Examples: {dup_ids[:5]}"
        )

    # 4. Check date correctness
    invalid_dates = df[df["order_date"] != date_str]["order_date"].unique().tolist()
    if invalid_dates:
        raise ValueError(
            f"[Validation Failed] Unexpected order_dates found for batch {date_str}: {invalid_dates}"
        )

    # 5. Check target labels
    unique_targets = set(df["late_delivery"].dropna().unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"[Validation Failed] Target 'late_delivery' contains invalid classes: {unique_targets}"
        )
    if df["late_delivery"].isna().any():
        raise ValueError(
            "[Validation Failed] Target 'late_delivery' contains null values!"
        )

    print(
        f"[Validation Passed] Gold data for {date_str} successfully validated ({len(df)} rows)."
    )
    return True


def run_pipeline_for_date(date_str: str) -> None:
    """Run full sequential pipeline for a single date."""
    print(f"=== Starting ETL Pipeline for date: {date_str} ===")
    extract_to_bronze(date_str)
    transform_to_silver(date_str)
    prepare_gold(date_str)
    validate_gold(date_str)
    print(f"=== Completed ETL Pipeline for date: {date_str} ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medallion ETL Pipeline")
    parser.add_argument(
        "--date", type=str, required=True, help="Processing date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["bronze", "silver", "gold", "validate", "all"],
        help="Pipeline step to run",
    )
    args = parser.parse_args()

    if args.step == "bronze":
        extract_to_bronze(args.date)
    elif args.step == "silver":
        transform_to_silver(args.date)
    elif args.step == "gold":
        prepare_gold(args.date)
    elif args.step == "validate":
        validate_gold(args.date)
    else:
        run_pipeline_for_date(args.date)
