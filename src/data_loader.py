"""
Data Loader — Supply Chain Disruption Intelligence Platform
Handles loading and initial validation of SCMS delivery history data.
"""

import pandas as pd
import numpy as np
import os


def load_raw_data(filepath: str = "data/SCMS_Delivery_History.csv") -> pd.DataFrame:
    """Load raw SCMS delivery history CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Download from Kaggle.")
    df = pd.read_csv(filepath, encoding="latin-1", low_memory=False)
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def load_cleaned_data(filepath: str = "data/SCMS_cleaned.csv") -> pd.DataFrame:
    """Load cleaned dataset from Phase 1."""
    df = pd.read_csv(filepath, low_memory=False)
    date_cols = [c for c in df.columns if "date" in c.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_features(filepath: str = "data/SCMS_features.csv") -> tuple:
    """Load feature matrix and target variable from Phase 2."""
    df = pd.read_csv(filepath)
    y = df["is_delayed"]
    X = df.drop(columns=["is_delayed"])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    print(f"Features: {X.shape[1]} | Samples: {len(X):,} | Delay rate: {y.mean()*100:.1f}%")
    return X, y


def validate_data(df: pd.DataFrame) -> dict:
    """Run basic data quality checks."""
    report = {
        "total_rows":     len(df),
        "total_cols":     df.shape[1],
        "missing_pct":    (df.isnull().sum().sum() / df.size * 100).round(2),
        "duplicate_rows": df.duplicated().sum(),
    }
    for k, v in report.items():
        print(f"  {k}: {v}")
    return report
