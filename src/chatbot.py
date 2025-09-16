
import pandas as pd

def answer_question(df, question):
    q = question.lower()
    df2 = df.copy()
    if 'spent' in q and 'month' in q:
        # try to extract month/year from question (simple)
        # default: last month
        df2['Date'] = pd.to_datetime(df2['Date'])
        last_month = df2['Date'].dt.to_period('M').max()
        total = df2[df2['Date'].dt.to_period('M')==last_month]['Amount'].sum()
        return f"Total spend for {last_month} is {total:.2f}"
    if 'how much' in q and 'food' in q:
        total = df2[df2['Category']=='Food']['Amount'].sum() if 'Category' in df2.columns else 0
        return f"Total spent on Food (from categorized data) is {total:.2f}"
    if 'top' in q and 'categories' in q:
        if 'Category' in df2.columns:
            summary = df2.groupby('Category')['Amount'].sum().sort_values(ascending=False).head(5)
            return "Top categories by spend:\n" + "\n".join([f"{i}: {v:.2f}" for i,v in summary.items()])
        else:
            return "Data is not categorized yet. Run classification first."
    return "Sorry, I could not understand. Try queries like: 'How much did I spend last month?', 'How much spent on Food?', 'Top categories'"
