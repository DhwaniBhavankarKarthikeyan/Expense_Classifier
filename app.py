
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.utils import load_csv, ensure_models_dir
from src import classifier, anomaly, forecast, chatbot

ensure_models_dir()

st.set_page_config(page_title="Expense Classifier v2", layout="wide")
st.title("Expense Classifier v2 - AIML Student Project Demo")

menu = st.sidebar.selectbox("Choose action", ["About", "Train Models (once)", "Upload & Classify", "Visualize & Detect Anomalies", "Forecast", "Chatbot"])

DATA_PATH = os.path.join("data","sample_transactions_labeled.csv")

if menu == "About":
    st.markdown("""
    **This enhanced demo includes:**
    - ML-based classifier (TF-IDF + Logistic Regression)
    - Anomaly detector (IsolationForest)
    - Simple monthly forecasting (Linear Regression)
    - Lightweight chatbot-like Q&A over your data

    **Instructions**
    1. If first time: go to 'Train Models (once)' and click Train.
    2. Then go to 'Upload & Classify' to upload your transactions CSV and classify.
    3. Use Visualize to see charts and anomalies.
    4. Use Forecast to see next month prediction.
    """)

if menu == "Train Models (once)":
    st.header("Train models using included labeled dataset")
    st.write(f"Training data path: {DATA_PATH}")
    if st.button("Train classifier and anomaly detector"):
        with st.spinner("Training classifier..."):
            report = classifier.train_and_save_model(DATA_PATH)
            st.success("Classifier trained and saved.")
            st.json(report)
        with st.spinner("Training anomaly detector..."):
            df = pd.read_csv(DATA_PATH)
            anomaly.train_anomaly(df)
            st.success("Anomaly detector trained and saved.")

if menu == "Upload & Classify":
    st.header("Upload transactions CSV (Date,Description,Amount)")
    uploaded = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded is not None:
        df = load_csv(uploaded)
        st.subheader("Raw data")
        st.dataframe(df.head(200))
        if st.button("Classify uploaded data"):
            try:
                dfc = classifier.predict_categories(df)
                st.success("Data classified.")
                st.dataframe(dfc.head(200))
                st.download_button("Download classified CSV", data=dfc.to_csv(index=False), file_name='classified_transactions.csv')
            except Exception as e:
                st.error(str(e))
                st.info("Train the model first via 'Train Models (once)'.")

if menu == "Visualize & Detect Anomalies":
    st.header("Visualize classified data and detect anomalies")
    uploaded = st.file_uploader("Upload classified CSV (or raw CSV)", type=['csv'], key='viz')
    if uploaded is not None:
        df = load_csv(uploaded)
        if 'Category' not in df.columns:
            st.info("Data is not categorized. You can classify it in 'Upload & Classify'. Proceeding with raw data.")
        st.subheader("Summary")
        st.write(df.describe(include='all'))
        # Pie chart if categorized
        if 'Category' in df.columns:
            cat_counts = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
            fig1, ax1 = plt.subplots()
            ax1.pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)
        # Anomaly detection
        if st.button("Detect anomalies (requires trained anomaly model)"):
            try:
                dfa = anomaly.detect(df)
                st.write("Anomalies flagged as True:")
                st.dataframe(dfa[dfa['anomaly_flag']==True])
            except Exception as e:
                st.error(str(e))
                st.info("Train the anomaly model first via 'Train Models (once)'.")

if menu == "Forecast":
    st.header("Monthly expense forecasting (simple linear regression)")
    uploaded = st.file_uploader("Upload CSV for forecasting (Date,Amount)", type=['csv'], key='forecast')
    if uploaded is not None:
        df = load_csv(uploaded)
        res = forecast.forecast_next_month(df)
        if res is None:
            st.info("Not enough history to forecast (need at least 3 months). Showing history only.")
            st.dataframe(df.head())
        else:
            st.subheader("Forecast result")
            st.write(f"Predicted next month total expense: {res['predicted_next_month_amount']:.2f}")
            st.subheader("History (monthly totals)")
            st.dataframe(res['history'])

if menu == "Chatbot":
    st.header("Lightweight Q&A over your transactions")
    uploaded = st.file_uploader("Upload classified CSV (recommended)", type=['csv'], key='chatbot')
    question = st.text_input("Ask a question (examples: 'How much did I spend last month?', 'How much spent on Food?')")
    if uploaded is not None and question:
        df = load_csv(uploaded)
        ans = chatbot.answer_question(df, question)
        st.write(ans)
    elif question:
        st.info("Upload your CSV so the chatbot can answer from your data.")

