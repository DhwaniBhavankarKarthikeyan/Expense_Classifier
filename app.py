import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objs as go
import requests

from src.financial_data import get_financial_data
from src.tickers import is_valid_ticker

# ---------------------------
# Expense Chatbot Functions
# ---------------------------
def build_context(df):
    summary = []
    total = df["Amount"].sum()
    summary.append(f"Total expenses: {total}")
    for cat, amt in df.groupby("Category")["Amount"].sum().items():
        summary.append(f"Spent {amt} on {cat}")
    return summary

def finance_chatbot(query, df):
    facts = build_context(df)
    documents = facts + [query]
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity([vectors[-1]], vectors[:-1])
    idx = cosine_sim.argmax()
    return facts[idx]

# ---------------------------
# Stock Tracker Functions
# ---------------------------
def plot_stock_data(ticker, data):
    trace = go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name=ticker,
    )
    layout = go.Layout(title=f"{ticker} Stock Data (3 Months)")
    fig = go.Figure(data=[trace], layout=layout)
    return fig

def summarize_financial_data(news, ticker, api_key):
    prompt = f'News of {ticker}: "{news}"\nIs {ticker} a buy at the moment? Provide a summary:\n'
    url = "https://api.gemini.com/v1/ai"  # replace if needed
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"prompt": prompt, "model": "gemini-1", "max_tokens": 1024, "temperature": 0.7}
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json().get("output_text", "No summary available.")
    else:
        return f"Error: {response.status_code} - {response.text}"

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="FinTech AI Suite", layout="wide")
st.title("💰 FinTech AI Suite")

# ---------------------------
# Tabs for Expense and Stock
# ---------------------------
main_tabs = st.tabs(["💰 Expense Management", "📈 Stock Tracker"])

# --------------------------- Expense Management Tab ---------------------------
with main_tabs[0]:
    uploaded_file = st.sidebar.file_uploader("📂 Upload your transactions CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "Amount" not in df.columns or "Category" not in df.columns:
            st.error("CSV must contain at least 'Amount' and 'Category' columns.")
        else:
            st.sidebar.success("✅ Data uploaded successfully!")

            sub_tabs = st.tabs(["📊 Classification", "📈 Visualization", "🔮 Forecasting", "🤖 Chatbot"])

            # Classification
            with sub_tabs[0]:
                st.header("📊 Expense Classification")
                st.write(df.head())
                category_totals = df.groupby("Category")["Amount"].sum()
                st.bar_chart(category_totals)

            # Visualization
            with sub_tabs[1]:
                st.header("📈 Expense Visualization")
                fig, ax = plt.subplots()
                df.groupby("Category")["Amount"].sum().plot(kind="pie", autopct="%1.1f%%", ax=ax)
                st.pyplot(fig)

            # Forecasting
            with sub_tabs[2]:
                st.header("🔮 Expense Forecasting (Simple)")
                if "Date" not in df.columns:
                    st.warning("CSV must have a 'Date' column for forecasting.")
                else:
                    monthly = df.groupby("Date")["Amount"].sum().reset_index()
                    st.line_chart(monthly.set_index("Date"))
                    if len(monthly) > 1:
                        avg = monthly["Amount"].mean()
                        st.success(f"📌 Projected next period expense: {round(avg,2)}")

            # Chatbot
            with sub_tabs[3]:
                st.header("🤖 Finance Chatbot")
                user_query = st.text_input("💬 Ask a question about your expenses:")
                if user_query:
                    answer = finance_chatbot(user_query, df)
                    st.success(answer)
    else:
        st.info("⬅️ Please upload a CSV file to begin.")

# --------------------------- Stock Tracker Tab ---------------------------
with main_tabs[1]:
    st.header("📈 Stock Analysis with Gemini AI")
    ticker = st.text_input("Enter stock ticker:", key="ticker_tab")
    gemini_api_key = st.text_input("Enter Gemini API Key:", type="password", key="gemini_key")

    if st.button("Get Summary 📊", key="ticker_btn"):
        if ticker and is_valid_ticker(ticker):
            data, news, stock_data = get_financial_data(ticker)

            # Stock Chart
            st.plotly_chart(plot_stock_data(ticker, stock_data))

            # Financial Metrics
            st.subheader("Financial Metrics")
            st.write("\n".join([f"- **{k}:** {v}" for k, v in data.items()]))

            # Gemini Summary
            if gemini_api_key:
                summary = summarize_financial_data(news, ticker, gemini_api_key)
                st.subheader("Gemini AI Summary")
                st.write(summary)
            else:
                st.warning("Enter your Gemini API Key to get AI summary.")
        else:
            st.warning("Please enter a valid stock ticker, e.g., MSFT.")
