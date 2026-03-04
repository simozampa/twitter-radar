"""
Apify Twitter scraper client.
Used for broad trend discovery without burning Twitter API credits.
Scrapes trending topics, search results, and high-engagement tweets.
"""

import time
import logging
from typing import Optional
import requests

logger = logging.getLogger("twitter_radar.apify")

# Popular Apify Twitter actors
ACTORS = {
    # Primary: apidojo tweet scraper v2
    "tweet_scraper": "apidojo/tweet-scraper",
    # Fallback: web.harvester search scraper
    "search_scraper": "web.harvester/easy-twitter-search-scraper",
    # User/profile scraper
    "user_scraper": "apidojo/twitter-user-scraper",
}

APIFY_BASE = "https://api.apify.com/v2"


class ApifyTwitterClient:
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        })

    def _run_actor(self, actor_id: str, input_data: dict,
                   timeout_secs: int = 120) -> list:
        """
        Run an Apify actor synchronously and return results.
        """
        # Actor ID format: username/actor-name → URL needs username~actor-name
        actor_url_id = actor_id.replace("/", "~")
        url = f"{APIFY_BASE}/acts/{actor_url_id}/runs"
        params = {
            "token": self.api_token,
            "timeout": timeout_secs,
            "waitForFinish": timeout_secs,
        }

        logger.info(f"Running Apify actor: {actor_id}")
        resp = self.session.post(url, params=params, json=input_data, timeout=timeout_secs + 30)

        if resp.status_code not in (200, 201):
            logger.error(f"Apify error {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()

        run_data = resp.json().get("data", {})
        status = run_data.get("status")

        if status != "SUCCEEDED":
            logger.warning(f"Actor run status: {status}")
            # Try to get partial results anyway
            if status not in ("SUCCEEDED", "RUNNING"):
                return []

        # Fetch results from default dataset
        dataset_id = run_data.get("defaultDatasetId")
        if not dataset_id:
            logger.error("No dataset ID in run response")
            return []

        return self._get_dataset(dataset_id)

    def _get_dataset(self, dataset_id: str, limit: int = 1000) -> list:
        """Fetch items from an Apify dataset."""
        url = f"{APIFY_BASE}/datasets/{dataset_id}/items"
        params = {
            "token": self.api_token,
            "limit": limit,
            "format": "json",
        }

        resp = self.session.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            logger.error(f"Dataset fetch error: {resp.status_code}")
            return []

        return resp.json()

    def search_tweets(self, queries: list[str], max_tweets: int = 200,
                      since_hours: float = 24, sort_by: str = "Top") -> list:
        """
        Search tweets using Apify tweet scraper.
        Uses Twitter advanced search syntax.
        Cost: ~$0.40 per 1,000 tweets.
        sort_by: "Top" or "Latest"
        """
        # Convert queries to Twitter advanced search format
        # Each query needs min 50 results on this actor
        search_terms = []
        for q in queries:
            # Add lang:en and min engagement filters
            search_terms.append(f"{q} lang:en min_faves:50")

        input_data = {
            "searchTerms": search_terms,
            "sort": sort_by,
            "maxItems": max_tweets,
        }

        # Try primary actor, fall back to secondary
        for actor_key in ["tweet_scraper", "search_scraper"]:
            actor_id = ACTORS[actor_key]
            try:
                raw_results = self._run_actor(actor_id, input_data)
                if raw_results:
                    return self._normalize_tweets(raw_results)
                logger.warning(f"Actor {actor_id} returned no results, trying next")
            except Exception as e:
                logger.warning(f"Apify actor {actor_id} failed: {e}, trying next")
                continue

        logger.error("All Apify actors failed for search")
        return []

    def get_trending_topics(self, woeid: int = 1) -> list:
        """
        Get Twitter trending topics.
        woeid: 1 = worldwide, 23424977 = US
        """
        input_data = {
            "woeid": woeid,
        }

        try:
            results = self._run_actor(ACTORS["trends"], input_data, timeout_secs=60)
        except Exception as e:
            logger.error(f"Apify trends failed: {e}")
            return []

        return results

    def scrape_user_tweets(self, usernames: list[str], max_tweets_per_user: int = 20,
                           since_hours: float = 24) -> list:
        """
        Scrape tweets from specific users via Apify.
        No Twitter API credits consumed.
        """
        # Build handles list
        handles = [u if u.startswith("@") else f"@{u}" for u in usernames]

        input_data = {
            "handles": handles,
            "maxTweets": max_tweets_per_user * len(handles),
            "sort": "Latest",
        }

        try:
            raw_results = self._run_actor(ACTORS["tweet_scraper"], input_data)
        except Exception as e:
            logger.error(f"Apify user scrape failed: {e}")
            return []

        return self._normalize_tweets(raw_results)

    def _normalize_tweets(self, raw_tweets: list) -> list:
        """
        Normalize Apify tweet format to our internal format.
        Different actors return slightly different schemas.
        """
        normalized = []

        for raw in raw_tweets:
            try:
                # Handle different Apify actor schemas
                tweet = {
                    "tweet_id": str(raw.get("id", raw.get("tweetId", raw.get("id_str", "")))),
                    "author_id": str(raw.get("author_id", raw.get("userId", ""))),
                    "author_username": (
                        raw.get("author", {}).get("userName", "") or
                        raw.get("username", "") or
                        raw.get("screen_name", "") or
                        raw.get("user", {}).get("screen_name", "")
                    ),
                    "author_followers": (
                        raw.get("author", {}).get("followers", 0) or
                        raw.get("user", {}).get("followers_count", 0)
                    ),
                    "text": (
                        raw.get("text", "") or
                        raw.get("full_text", "") or
                        raw.get("tweetText", "")
                    ),
                    "created_at": (
                        raw.get("createdAt", "") or
                        raw.get("created_at", "") or
                        raw.get("date", "")
                    ),
                    "likes": (
                        raw.get("likeCount", 0) or
                        raw.get("favorite_count", 0) or
                        raw.get("likes", 0)
                    ),
                    "retweets": (
                        raw.get("retweetCount", 0) or
                        raw.get("retweet_count", 0) or
                        raw.get("retweets", 0)
                    ),
                    "replies": (
                        raw.get("replyCount", 0) or
                        raw.get("reply_count", 0) or
                        raw.get("replies", 0)
                    ),
                    "quotes": (
                        raw.get("quoteCount", 0) or
                        raw.get("quote_count", 0) or
                        0
                    ),
                    "impressions": (
                        raw.get("viewCount", 0) or
                        raw.get("views", 0) or
                        0
                    ),
                    "url": raw.get("url", raw.get("tweetUrl", "")),
                    "source": "apify",
                }

                # Skip if no text or ID
                if tweet["text"] and tweet["tweet_id"]:
                    normalized.append(tweet)

            except Exception as e:
                logger.debug(f"Failed to normalize tweet: {e}")
                continue

        logger.info(f"Normalized {len(normalized)}/{len(raw_tweets)} tweets from Apify")
        return normalized
