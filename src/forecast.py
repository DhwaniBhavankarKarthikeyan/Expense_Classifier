
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def monthly_aggregate(df):
    df2 = df.copy()
    df2['Date'] = pd.to_datetime(df2['Date'])
    df2['month'] = df2['Date'].dt.to_period('M').astype(str)
    agg = df2.groupby('month')['Amount'].sum().reset_index()
    agg['month_index'] = range(len(agg))
    return agg

def forecast_next_month(df, months_ahead=1):
    agg = monthly_aggregate(df)
    if len(agg) < 3:
        return None
    X = agg[['month_index']].values
    y = agg['Amount'].values
    model = LinearRegression()
    model.fit(X, y)
    next_idx = np.array([[agg['month_index'].max() + months_ahead]])
    pred = model.predict(next_idx)[0]
    return {'predicted_next_month_amount': float(pred), 'history': agg}
