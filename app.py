import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Lightweight Chatbot
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
# Streamlit App
# ---------------------------
st.set_page_config(page_title="FinTech Expense Classifier", layout="wide")
st.title("💰 FinTech AI Application")

# Sidebar upload (only once)
uploaded_file = st.sidebar.file_uploader("📂 Upload your transactions CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Ensure expected columns exist
    if "Amount" not in df.columns or "Category" not in df.columns:
        st.error("CSV must contain at least 'Amount' and 'Category' columns.")
    else:
        st.sidebar.success("✅ Data uploaded successfully!")

        # Tabs
        tabs = st.tabs(["📊 Classification", "📈 Visualization", "🔮 Forecasting", "🤖 Chatbot"])

        # ---------------- Classification ----------------
        with tabs[0]:
            st.header("📊 Expense Classification")
            st.write(df.head())

            category_totals = df.groupby("Category")["Amount"].sum()
            st.bar_chart(category_totals)

        # ---------------- Visualization ----------------
        with tabs[1]:
            st.header("📈 Expense Visualization")

            fig, ax = plt.subplots()
            df.groupby("Category")["Amount"].sum().plot(kind="pie", autopct="%1.1f%%", ax=ax)
            st.pyplot(fig)

        # ---------------- Forecasting ----------------
        with tabs[2]:
            st.header("🔮 Expense Forecasting (Simple)")

            if "Date" not in df.columns:
                st.warning("CSV must have a 'Date' column for forecasting.")
            else:
                monthly = df.groupby("Date")["Amount"].sum().reset_index()
                st.line_chart(monthly.set_index("Date"))

                if len(monthly) > 1:
                    avg = monthly["Amount"].mean()
                    st.success(f"📌 Projected next period expense: {round(avg,2)}")

        # ---------------- Chatbot ----------------
        with tabs[3]:
            st.header("🤖 Finance Chatbot")
            st.write("Ask me about your expenses (e.g., *total expense, food spending, travel costs*)")

            user_query = st.text_input("💬 Your question:")
            if user_query:
                answer = finance_chatbot(user_query, df)
                st.success(answer)
else:
    st.info("⬅️ Please upload a CSV file to begin.")
