"""
Model — Supply Chain Disruption Intelligence Platform
XGBoost training, evaluation, and risk scoring utilities.
"""

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             f1_score, classification_report)


def train_xgboost(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Train XGBoost model with early stopping."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    delay_rate = y_train.mean()
    scale_pos_weight = (1 - delay_rate) / delay_rate if delay_rate < 0.4 else 1.0

    model = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42, eval_metric="auc",
        early_stopping_rounds=30, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model, X_test, y_test


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return full evaluation metrics."""
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "auc_roc":   roc_auc_score(y_test, y_pred_prob),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
    }
    print("=" * 45)
    print("  MODEL PERFORMANCE")
    print("=" * 45)
    for k, v in metrics.items():
        print(f"  {k:12}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=["On-Time", "Delayed"]))
    return metrics


def score_shipments(model, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
    """Apply risk scoring to all shipments."""
    proba = model.predict_proba(X)[:, 1]
    risk_df = pd.DataFrame({
        "delay_probability": proba,
        "risk_score":        (proba * 100).round(1),
    })
    risk_df["risk_tier"] = pd.cut(
        risk_df["delay_probability"],
        bins=[0, 0.30, 0.60, 0.80, 1.01],
        labels=["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    )
    if y is not None:
        risk_df["actual_delayed"] = y.values
    return risk_df


def save_model(model, path: str = "outputs/models/xgboost_delay_model.pkl") -> None:
    """Save trained model to disk."""
    joblib.dump(model, path)
    print(f"Model saved: {path}")


def load_model(path: str = "outputs/models/xgboost_delay_model.pkl"):
    """Load trained model from disk."""
    return joblib.load(path)
