
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "clf_pipeline.joblib")

def train_and_save_model(labeled_csv_path):
    df = pd.read_csv(labeled_csv_path)
    df = df.dropna(subset=['Description','Category'])
    X = df['Description']
    y = df['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=2000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    joblib.dump(pipeline, MODEL_PATH)
    return report

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def predict_categories(df):
    # expects dataframe with Description column
    model = load_model()
    if model is None:
        raise RuntimeError("Model not found. Train the model first.")
    preds = model.predict(df['Description'])
    df = df.copy()
    df['Category'] = preds
    return df
