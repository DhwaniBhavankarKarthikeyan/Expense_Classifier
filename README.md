# Expense Classifier v2 - AIML Final Year Project Level Demo

This project is an enhanced FinTech app (Streamlit) with:
- ML-based expense classification (TF-IDF + LogisticRegression)
- Anomaly detection (IsolationForest) to flag suspicious transactions
- Simple monthly expense forecasting (Linear Regression)
- A lightweight chatbot-like Q&A that answers questions by querying the data

## How to run

1. Unzip and `cd fintech_expense_classifier_v2`
2. Create and activate a virtualenv (recommended)
   ```
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate    # Windows
   ```
3. Install dependencies
   ```
   pip install -r requirements.txt
   ```
4. Run the app
   ```
   streamlit run app.py
   ```

Open http://localhost:8501

## Files
- app.py : Streamlit app
- src/ : helper modules (classifier, anomaly, forecast, utils)
- data/sample_transactions_labeled.csv : synthetic labeled data used to train the classifier
- models/ : where trained models are saved automatically

