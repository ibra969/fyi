# preprocessing.py
# Section 4.2 – normalize, clean, tokenize

import json
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

def clean_text(text: str) -> str:
    # remove URLs, punctuation, digits
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

def preprocess_tweets(raw_json_path: str, output_csv: str):
    df = pd.read_json(raw_json_path)
    df["clean"] = df["text"].apply(clean_text)
    df["tokens"] = df["clean"].apply(tokenizer.tokenize)
    df["tokens"] = df["tokens"].apply(lambda toks: [t for t in toks if t not in STOPWORDS and len(t)>2])
    df.to_csv(output_csv, index=False)
    return df

if __name__ == "__main__":
    df = preprocess_tweets("raw_tweets.json", "tweets_preprocessed.csv")
    print(f"Preprocessed {len(df)} tweets.")
