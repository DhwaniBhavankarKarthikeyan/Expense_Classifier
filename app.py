import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# --------------------------
# Helper Functions
# --------------------------

def train_models():
    data = pd.read_csv("data/sample_transactions_labeled.csv")
    X = data["Description"]
    y = data["Category"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_vec, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    joblib.dump(clf, "models/classifier.pkl")

    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(data[["Amount"]])
    joblib.dump(iso, "models/anomaly.pkl")

def load_models():
    vectorizer = joblib.load("models/vectorizer.pkl")
    clf = joblib.load("models/classifier.pkl")
    iso = joblib.load("models/anomaly.pkl")
    return vectorizer, clf, iso

def classify_data(df, vectorizer, clf, iso):
    X_new = vectorizer.transform(df["Description"])
    df["PredictedCategory"] = clf.predict(X_new)
    df["Anomaly"] = iso.predict(df[["Amount"]])
    df["Anomaly"] = df["Anomaly"].map({1: "Normal", -1: "Suspicious"})
    return df

def forecast_expenses(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    monthly = df.groupby("Month")["Amount"].sum().reset_index()

    monthly["t"] = range(len(monthly))
    X = monthly[["t"]]
    y = monthly["Amount"]

    model = LinearRegression()
    model.fit(X, y)

    next_t = [[len(monthly)]]
    forecast = model.predict(next_t)[0]

    return monthly, forecast

def chatbot_query(df, query):
    query = query.lower()
    response = "Sorry, I couldn’t understand the question."

    if "total" in query:
        response = f"Your total spending is {df['Amount'].sum():.2f}."
    elif "food" in query:
        total = df[df["PredictedCategory"] == "Food"]["Amount"].sum()
        response = f"Your spending on Food is {total:.2f}."
    elif "last month" in query:
        df["Date"] = pd.to_datetime(df["Date"])
        last_month = df["Date"].dt.to_period("M").max()
        total = df[df["Date"].dt.to_period("M") == last_month]["Amount"].sum()
        response = f"Your spending in {last_month} was {total:.2f}."
    elif "top" in query and "category" in query:
        top_cat = df.groupby("PredictedCategory")["Amount"].sum().idxmax()
        response = f"Your top spending category is {top_cat}."

    return response

# --------------------------
# Streamlit App
# --------------------------

st.set_page_config(page_title="AI FinTech Expense Classifier", layout="wide")
st.title("💸 AI FinTech Expense Classifier")

# Sidebar Upload (only once)
uploaded_file = st.sidebar.file_uploader("Upload your transactions CSV", type=["csv"])

# Train models once (if not already trained)
if not os.path.exists("models/classifier.pkl"):
    train_models()

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    vectorizer, clf, iso = load_models()
    df = classify_data(df, vectorizer, clf, iso)
    st.session_state["classified_data"] = df

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🧾 Data", "📊 Visualization", "📈 Forecast", "🤖 Chatbot"])

with tab1:
    st.header("🧾 Classified Transactions")
    if "classified_data" in st.session_state:
        st.dataframe(st.session_state["classified_data"].head())
    else:
        st.info("Upload a CSV file to see classified transactions.")

with tab2:
    st.header("📊 Visualization & Anomalies")
    if "classified_data" in st.session_state:
        df = st.session_state["classified_data"]

        fig, ax = plt.subplots()
        df["PredictedCategory"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        st.pyplot(fig)

        st.subheader("🚨 Suspicious Transactions")
        st.dataframe(df[df["Anomaly"] == "Suspicious"])
    else:
        st.info("Upload a CSV first.")

with tab3:
    st.header("📈 Expense Forecast")
    if "classified_data" in st.session_state:
        df = st.session_state["classified_data"]
        monthly, forecast = forecast_expenses(df)

        st.line_chart(monthly.set_index("Month")["Amount"])
        st.success(f"Predicted spending for next month: {forecast:.2f}")
    else:
        st.info("Upload a CSV first.")

with tab4:
    st.header("🤖 Finance Chatbot")
    if "classified_data" in st.session_state:
        df = st.session_state["classified_data"]
        query = st.text_input("Ask me about your expenses:")
        if query:
            st.write("💬", chatbot_query(df, query))
    else:
        st.info("Upload a CSV first.")
