# sentiment_analysis.py
# Section 6 – VADER & LIWC sentiment

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import liwc

# initialize VADER
sia = SentimentIntensityAnalyzer()

# load LIWC dictionary
liwc_parser = liwc.load_token_parser("LIWC2007_English.dic")

def vader_scores(text: str) -> dict:
    return sia.polarity_scores(text)

def liwc_counts(tokens: list) -> dict:
    counts = {}
    for token in tokens:
        for category in liwc_parser(token):
            counts[category] = counts.get(category, 0) + 1
    return counts

def analyze_sentiment(preprocessed_csv: str):
    df = pd.read_csv(preprocessed_csv)
    df["vader"] = df["clean"].apply(vader_scores)
    df = pd.concat([df.drop(columns=["vader"]), df["vader"].apply(pd.Series)], axis=1)
    df["liwc"] = df["tokens"].apply(eval).apply(liwc_counts)
    # expand LIWC categories into columns
    liwc_df = df["liwc"].apply(pd.Series).fillna(0).astype(int)
    df = pd.concat([df.drop(columns=["liwc"]), liwc_df], axis=1)
    df.to_csv("sentiment_analysis.csv", index=False)
    return df

if __name__ == "__main__":
    df = analyze_sentiment("tweets_preprocessed.csv")
    print("Completed sentiment analysis.")
