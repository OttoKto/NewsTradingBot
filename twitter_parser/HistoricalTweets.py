import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from paths import get_historical_tweets_path


def fetch_historical_tweets(
        accounts: List[str],
        days: int,
        api_key: str,
        max_results_per_request: int = 100
) -> List[Dict]:
    """
    Fetches historical original tweets (excluding replies and retweets) from specified accounts
    within a given number of days using TwitterAPI.io.

    Args:
        accounts (List[str]): List of Twitter usernames (without @)
        days (int): Number of days to look back for tweets
        api_key (str): TwitterAPI.io API key for authentication
        max_results_per_request (int): Maximum number of tweets per API request (up to 500)

    Returns:
        List[Dict]: List of unique original tweets from specified accounts

    Notes:
        - Requires TwitterAPI.io Pro API Access for historical data beyond 7 days
        - Handles pagination and rate limiting
        - Deduplicates tweets based on tweet ID
        - Excludes replies (isReply=True) and retweets (non-empty retweeted_tweet)
        - Uses /twitter/tweet/advanced_search endpoint
    """
    base_url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
    headers = {"x-api-key": api_key}
    all_tweets = []
    seen_tweet_ids = set()
    max_retries = 3

    # Remove duplicates from accounts list
    accounts = list(set(accounts))

    # Calculate time range
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    for account in accounts:
        print(f"Fetching tweets for @{account}...")
        query = f"from:{account} since:{start_time_str} until:{end_time_str}"
        cursor = None

        while True:
            # Prepare query parameters
            params = {
                "query": query,
                "queryType": "Latest",
                "maxResults": max_results_per_request
            }
            if cursor:
                params["cursor"] = cursor

            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Make API request
                    response = requests.get(base_url, headers=headers, params=params)
                    if response.status_code == 401:
                        print(f"401 Unauthorized error for @{account}. Check API key validity and permissions.")
                        print("Ensure you have Pro API access for historical searches. Visit https://x.ai/api for details.")
                        return all_tweets  # Exit on 401 to avoid unnecessary retries

                    response.raise_for_status()  # Raise exception for other bad status codes
                    data = response.json()

                    # Extract tweets and metadata
                    tweets = data.get("tweets", [])
                    has_next_page = data.get("has_next_page", False)
                    cursor = data.get("next_cursor", None)

                    # Filter out replies, retweets, and duplicates
                    new_tweets = [
                        tweet for tweet in tweets
                        if tweet.get("id") not in seen_tweet_ids
                        and not tweet.get("isReply", False)  # Exclude replies
                        and not tweet.get("retweeted_tweet")  # Exclude retweets (non-empty retweeted_tweet)
                    ]

                    # Process new tweets
                    for tweet in new_tweets:
                        seen_tweet_ids.add(tweet.get("id"))
                        all_tweets.append({
                            "id": tweet.get("id"),
                            "text": tweet.get("text"),
                            "created_at": tweet.get("createdAt"),
                            "username": account,
                            "metadata": {
                                "retweet_count": tweet.get("retweetCount"),
                                "reply_count": tweet.get("replyCount"),
                                "like_count": tweet.get("likeCount"),
                                "quote_count": tweet.get("quoteCount"),
                                "view_count": tweet.get("viewCount"),
                                "lang": tweet.get("lang"),
                                "bookmark_count": tweet.get("bookmarkCount")
                            }
                        })

                    # Break if no new tweets or no next page
                    if not new_tweets and not has_next_page:
                        break

                    # Continue with next page if available
                    if has_next_page:
                        break

                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        print(f"Failed to fetch tweets for @{account} after {max_retries} attempts: {str(e)}")
                        break

                    if response.status_code == 429:
                        print(f"Rate limit reached for @{account}. Waiting for 60 seconds...")
                        time.sleep(60)  # Wait for rate limit reset
                    else:
                        print(f"Error for @{account}: {str(e)}. Retrying {retry_count}/{max_retries}")
                        time.sleep(2 ** retry_count)  # Exponential backoff

            # Break outer loop if no more pages or no new tweets
            if not has_next_page and not new_tweets:
                break

    return all_tweets


def save_tweets_to_csv(tweets: List[Dict], filename: str):
    """
    Saves tweets to a CSV file.

    Args:
        tweets (List[Dict]): List of tweet dictionaries
        filename (str): Output CSV filename
    """
    if tweets:
        df = pd.DataFrame(tweets)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Saved {len(tweets)} tweets to {filename}")
    else:
        print("No tweets to save")


# Example usage
if __name__ == "__main__":
    api_key = "d13e7230ecab4060871c64f8f3501413"  # Replace with your TwitterAPI.io API key
    #accounts = [
    #    "elonmusk", "cz_binance", "VitalikButerin", "SBF_FTX", "woonomic",
    #    "tyler", "twobitidiot", "crypto_birb", "TheCryptoDog", "filbfilb",
    #    "nayibbukele", "LynAldenContact", "pomp", "IOHK_Charles", "haydenzadams",
    #    "ErikVoorhees", "BarrySilbert", "naval", "katherineykwu", "fubar",
    #    "WuBlockchain", "CryptoCobain", "DylanLeClair_", "100trillionUSD",
    #    "CoinDesk", "CryptoHayes", "sassal0x", "APompliano", "michael_saylor", "balajis"
    #]
    accounts = ["woonomic", "WuBlockchain", "CoinDesk", "Cointelegraph"
    ]
    days = 720  # Number of days to look back

    tweets = fetch_historical_tweets(accounts, days, api_key)
    save_tweets_to_csv(tweets, get_historical_tweets_path())

    print(f"Fetched {len(tweets)} unique original tweets from {len(accounts)} accounts")