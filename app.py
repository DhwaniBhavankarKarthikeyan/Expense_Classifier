import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Dynamic category mapping
# ---------------------------
# Minimal keywords for generalization
CATEGORY_KEYWORDS = {
    "Travel": ["bus", "train", "uber", "cab", "flight", "metro", "taxi"],
    "Entertainment": ["netflix", "prime", "spotify", "movie", "cinema", "theater", "concert"],
    "Food": ["domino", "pizza", "restaurant", "groceries", "mcdonald", "kfc", "cafe"],
    "Shopping": ["amazon", "flipkart", "mall", "shopping", "store", "clothes", "apparel"],
}

def map_to_category(item):
    item_lower = str(item).lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k in item_lower for k in keywords):
            return cat
    return "Other"

def apply_dynamic_categories(df):
    # Create a standardized category column
    df['Category_Std'] = df['Category'].apply(map_to_category)
    return df

# ---------------------------
# Finance Q&A
# ---------------------------
def get_total_expense(df):
    return f"Your total expense is {df['Amount'].sum():.2f}"

def get_category_expense(df, category):
    total = df[df['Category_Std'] == category]['Amount'].sum()
    return f"You spent {total:.2f} on {category}"

def get_average_expense(df):
    return f"Your average expense per transaction is {df['Amount'].mean():.2f}"

def get_max_expense(df):
    idx = df['Amount'].idxmax()
    row = df.loc[idx]
    return f"Your largest expense is {row['Amount']:.2f} in category {row['Category_Std']}"

def get_min_expense(df):
    idx = df['Amount'].idxmin()
    row = df.loc[idx]
    return f"Your smallest expense is {row['Amount']:.2f} in category {row['Category_Std']}"

def get_category_percentage(df, category):
    total = df['Amount'].sum()
    cat_total = df[df['Category_Std'] == category]['Amount'].sum()
    percent = (cat_total / total * 100) if total else 0
    return f"{category} makes up {percent:.1f}% of your total expenses"

def get_most_expensive_category(df):
    totals = df.groupby('Category_Std')['Amount'].sum()
    cat = totals.idxmax()
    return f"The category with the highest total expense is {cat} ({totals[cat]:.2f})"

def get_least_expensive_category(df):
    totals = df.groupby('Category_Std')['Amount'].sum()
    cat = totals.idxmin()
    return f"The category with the lowest total expense is {cat} ({totals[cat]:.2f})"

def get_transaction_count(df):
    return f"You have {len(df)} transactions in total"

def get_category_count(df):
    return f"You have {df['Category_Std'].nunique()} different categories"

def get_average_category_expense(df, category):
    cat_total = df[df['Category_Std'] == category]['Amount'].sum()
    count = len(df[df['Category_Std'] == category])
    avg = cat_total / count if count else 0
    return f"Average expense in {category} is {avg:.2f}"

# Map of question -> function (with optional category)
QUESTION_MAP = {
    "Total expense": lambda df: get_total_expense(df),
    "Average expense per transaction": lambda df: get_average_expense(df),
    "Transaction count": lambda df: get_transaction_count(df),
    "Number of categories": lambda df: get_category_count(df),
    "Largest single expense": lambda df: get_max_expense(df),
    "Smallest single expense": lambda df: get_min_expense(df),
    "Category with highest total expense": lambda df: get_most_expensive_category(df),
    "Category with lowest total expense": lambda df: get_least_expensive_category(df),
    "Percentage spent on Food": lambda df: get_category_percentage(df, "Food"),
    "Percentage spent on Travel": lambda df: get_category_percentage(df, "Travel"),
    "Percentage spent on Entertainment": lambda df: get_category_percentage(df, "Entertainment"),
    "Average expense in Food": lambda df: get_average_category_expense(df, "Food"),
    "Average expense in Travel": lambda df: get_average_category_expense(df, "Travel"),
    "Average expense in Entertainment": lambda df: get_average_category_expense(df, "Entertainment"),
    "Total spent on Shopping": lambda df: get_category_expense(df, "Shopping"),
}

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="FinTech Expense Classifier", layout="wide")
st.title("💰 FinTech AI Application")

uploaded_file = st.sidebar.file_uploader("📂 Upload your transactions CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validate columns
    if "Amount" not in df.columns or "Category" not in df.columns:
        st.error("CSV must contain at least 'Amount' and 'Category' columns.")
    else:
        st.sidebar.success("✅ Data uploaded successfully!")
        
        # Apply dynamic category mapping
        df = apply_dynamic_categories(df)

        # Tabs
        tabs = st.tabs(["📊 Classification", "📈 Visualization", "🔮 Forecasting", "🤖 Finance Q&A"])

        # ---------------- Classification ----------------
        with tabs[0]:
            st.header("📊 Expense Classification")
            st.write(df.head())
            category_totals = df.groupby("Category_Std")["Amount"].sum()
            st.bar_chart(category_totals)

        # ---------------- Visualization ----------------
        with tabs[1]:
            st.header("📈 Expense Visualization")
            category_totals = df.groupby("Category_Std")["Amount"].sum()
            total_amount = category_totals.sum()
            
            # Combine categories below 3% into 'Other'
            threshold = 0.03  # 3%
            large_categories = category_totals[category_totals / total_amount > threshold]
            small_categories = category_totals[category_totals / total_amount <= threshold]
            
            if not small_categories.empty:
                large_categories["Other"] = small_categories.sum()
            
            fig, ax = plt.subplots()
            large_categories.plot(kind="pie", autopct="%1.1f%%", ax=ax, startangle=90)
            ax.set_ylabel("")  # remove y-label for better visuals
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

        # ---------------- Finance Q&A ----------------
        with tabs[3]:
            st.header("🤖 Finance Q&A")
            st.write("Select a question from the dropdown to see your answer:")
            question = st.selectbox("💬 Choose a question:", list(QUESTION_MAP.keys()))
            if question:
                answer = QUESTION_MAP[question](df)
                st.success(answer)
else:
    st.info("⬅️ Please upload a CSV file to begin.")
