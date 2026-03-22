"""
Feature Engineering — Supply Chain Disruption Intelligence Platform
Transforms raw shipment data into ML-ready features.
"""

import pandas as pd
import numpy as np


def add_temporal_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add time-based features from order date."""
    df["order_month"]       = df[date_col].dt.month
    df["order_quarter"]     = df[date_col].dt.quarter
    df["order_dayofweek"]   = df[date_col].dt.dayofweek
    df["order_year"]        = df[date_col].dt.year
    df["is_end_of_quarter"] = df["order_month"].isin([3, 6, 9, 12]).astype(int)
    df["is_weekend_order"]  = (df["order_dayofweek"] >= 5).astype(int)
    return df


def add_vendor_risk_features(df: pd.DataFrame, vendor_col: str) -> pd.DataFrame:
    """Encode historical vendor delay rate as risk score."""
    vendor_stats = df.groupby(vendor_col).agg(
        vendor_delay_rate=("is_delayed", "mean"),
        vendor_shipment_count=("is_delayed", "count"),
    ).reset_index()
    df = df.merge(vendor_stats, on=vendor_col, how="left")
    df["vendor_risk_tier"] = pd.qcut(
        df["vendor_delay_rate"].fillna(df["vendor_delay_rate"].median()),
        q=3, labels=["low_risk", "medium_risk", "high_risk"]
    )
    return df


def add_composite_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Build composite risk score from vendor, country, and mode risk."""
    components = []
    for col, weight in [("vendor_delay_rate", 0.4),
                        ("country_delay_rate", 0.35),
                        ("mode_delay_rate", 0.25)]:
        if col in df.columns:
            norm = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-9)
            components.append(norm * weight)
    if components:
        df["composite_risk_score"] = sum(components)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select and return final feature matrix for modelling."""
    feature_cols = [
        "log_freight_cost", "freight_vs_median_ratio", "is_high_cost_shipment",
        "log_weight", "vendor_delay_rate", "vendor_shipment_count", "is_major_vendor",
        "country_delay_rate", "country_shipment_vol", "mode_delay_rate",
        "composite_risk_score", "order_month", "order_quarter",
        "order_dayofweek", "is_end_of_quarter", "is_weekend_order",
    ]
    available = [c for c in feature_cols if c in df.columns]
    mode_dummies = [c for c in df.columns if c.startswith("mode_") and c != "mode_delay_rate"]
    all_features = available + mode_dummies
    X = df[all_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    return X
