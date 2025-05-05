# topic_modeling.py
# Section 5 – BERTopic clustering

import pandas as pd
from bertopic import BERTopic

def run_topic_modeling(preprocessed_csv: str, num_topics: int = 10):
    df = pd.read_csv(preprocessed_csv)
    docs = df["clean"].tolist()
    topic_model = BERTopic(nr_topics=num_topics)
    topics, probs = topic_model.fit_transform(docs)
    df["topic"] = topics
    topic_model.get_topic_info().to_csv("topic_info.csv", index=False)
    df.to_csv("tweets_with_topics.csv", index=False)
    return topic_model

if __name__ == "__main__":
    model = run_topic_modeling("tweets_preprocessed.csv", num_topics=12)
    print("Saved topic assignments and info.")
