# data_collection.py
# Section 4.1 – fetch tweets via Twitter API

import os
import tweepy
import json
from datetime import datetime, timedelta

# load credentials from environment variables
API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET = os.getenv("TWITTER_API_SECRET")
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

client = tweepy.Client(bearer_token=BEARER_TOKEN)

def fetch_tweets(query: str, start_date: str, end_date: str, max_results: int = 100):
    """
    Fetch tweets matching `query` between start_date and end_date.
    Dates in 'YYYY-MM-DD' format.
    """
    tweets = []
    # Twitter API v2 pagination
    paginator = tweepy.Paginator(
        client.search_all_tweets,
        query=query,
        start_time=start_date + "T00:00:00Z",
        end_time=end_date + "T00:00:00Z",
        tweet_fields=["id","text","created_at","public_metrics","lang"],
        max_results=500,
    )
    for tweet_page in paginator:
        tweets.extend(tweet_page.data or [])
        if len(tweets) >= max_results:
            break
    return tweets[:max_results]

if __name__ == "__main__":
    # Example: fetch blockchain tweets from April–May 2024
    tweets = fetch_tweets("blockchain OR #blockchain", "2024-04-01", "2024-05-01", max_results=10000)
    # serialize to JSON
    with open("raw_tweets.json", "w", encoding="utf-8") as f:
        json.dump([t.data for t in tweets], f, ensure_ascii=False, indent=2)
    print(f"Fetched {len(tweets)} tweets.")
