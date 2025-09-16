
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "anomaly_if.joblib")

def train_anomaly(df):
    # Use Amount and description length as features
    X = df[['Amount']].copy()
    X['desc_len'] = df['Description'].astype(str).apply(len)
    model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    model.fit(X)
    joblib.dump(model, MODEL_PATH)
    return

def load_anomaly():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def detect(df):
    model = load_anomaly()
    if model is None:
        raise RuntimeError("Anomaly model not found. Train it first.")
    X = df[['Amount']].copy()
    X['desc_len'] = df['Description'].astype(str).apply(len)
    preds = model.predict(X)  # -1 for anomaly, 1 for normal
    df = df.copy()
    df['anomaly'] = preds
    df['anomaly_flag'] = df['anomaly'].apply(lambda x: True if x==-1 else False)
    return df
