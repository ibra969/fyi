# predictive_modeling.py
# Section 8 – regress engagement on sentiment & topics

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def run_models(sentiment_csv: str):
    df = pd.read_csv(sentiment_csv)
    # features: VADER compound + topic dummies
    X = df[["compound"]].join(pd.get_dummies(df["topic"], prefix="topic"))
    y = df["public_metrics.retweet_count"] + df["public_metrics.like_count"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # OLS
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    y_pred_ols = ols.predict(X_test)
    mse_ols = mean_squared_error(y_test, y_pred_ols)
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    # report
    print(f"OLS MSE: {mse_ols:.2f}")
    print(f"RF  MSE: {mse_rf:.2f}")
    # save models if desired
    return ols, rf

if __name__ == "__main__":
    run_models("sentiment_analysis.csv")
