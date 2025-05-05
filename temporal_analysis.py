# temporal_analysis.py
# Section 7 – time-series of tweet volumes & sentiment

import pandas as pd
import matplotlib.pyplot as plt

def plot_time_series(sentiment_csv: str):
    df = pd.read_csv(sentiment_csv, parse_dates=["created_at"])
    df.set_index("created_at", inplace=True)
    # aggregate daily counts & avg sentiment
    daily = df.resample("D").agg({
        "id": "count",
        "compound": "mean"
    }).rename(columns={"id": "tweet_count", "compound": "avg_sentiment"})
    daily.to_csv("daily_trends.csv")
    # plot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(daily.index, daily["tweet_count"])
    ax2.plot(daily.index, daily["avg_sentiment"])
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Tweet Count")
    ax2.set_ylabel("Avg. VADER Sentiment")
    fig.tight_layout()
    fig.savefig("temporal_trends.png")
    print("Saved temporal trends plot.")

if __name__ == "__main__":
    plot_time_series("sentiment_analysis.csv")
