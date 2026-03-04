"""
Twitter API v2 client for Trend Radar.
Handles authentication, search, rate limiting.
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
import requests

logger = logging.getLogger("twitter_radar.client")

API_BASE = "https://api.twitter.com/2"


class RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, max_requests: int, window_seconds: int = 900):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: list[float] = []

    def wait_if_needed(self):
        now = time.time()
        # Purge old entries
        self.requests = [t for t in self.requests if now - t < self.window_seconds]
        if len(self.requests) >= self.max_requests:
            sleep_time = self.requests[0] + self.window_seconds - now + 1
            if sleep_time > 0:
                logger.info(f"Rate limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        self.requests.append(time.time())


class TwitterClient:
    def __init__(self, bearer_token: str, rate_limit: int = 280):
        self.bearer_token = bearer_token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {bearer_token}",
            "User-Agent": "TwitterRadar/1.0"
        })
        self.limiter = RateLimiter(rate_limit, 900)

    def _get(self, endpoint: str, params: dict) -> dict:
        """Make a rate-limited GET request."""
        self.limiter.wait_if_needed()
        url = f"{API_BASE}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=30)

        if resp.status_code == 429:
            reset = int(resp.headers.get("x-rate-limit-reset", time.time() + 60))
            sleep_time = max(reset - time.time() + 1, 1)
            logger.warning(f"429 rate limited. Sleeping {sleep_time:.0f}s")
            time.sleep(sleep_time)
            return self._get(endpoint, params)

        if resp.status_code != 200:
            logger.error(f"API error {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()

        return resp.json()

    def search_recent(
        self,
        query: str,
        max_results: int = 100,
        since_hours: float = 2,
        next_token: Optional[str] = None
    ) -> dict:
        """
        Search recent tweets (last 7 days max on Pro plan).
        Returns dict with 'tweets' list and optional 'next_token'.
        """
        start_time = datetime.now(timezone.utc) - timedelta(hours=since_hours)

        params = {
            "query": query,
            "max_results": min(max_results, 100),
            "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tweet.fields": "created_at,public_metrics,author_id,conversation_id,context_annotations,entities",
            "user.fields": "username,public_metrics,verified",
            "expansions": "author_id",
            "sort_order": "relevancy"
        }

        if next_token:
            params["next_token"] = next_token

        data = self._get("tweets/search/recent", params)

        # Parse users into a lookup
        users = {}
        if "includes" in data and "users" in data["includes"]:
            for u in data["includes"]["users"]:
                users[u["id"]] = u

        tweets = []
        for t in data.get("data", []):
            metrics = t.get("public_metrics", {})
            author = users.get(t["author_id"], {})
            tweets.append({
                "tweet_id": t["id"],
                "author_id": t["author_id"],
                "author_username": author.get("username", "unknown"),
                "author_followers": author.get("public_metrics", {}).get("followers_count", 0),
                "text": t["text"],
                "created_at": t["created_at"],
                "likes": metrics.get("like_count", 0),
                "retweets": metrics.get("retweet_count", 0),
                "replies": metrics.get("reply_count", 0),
                "quotes": metrics.get("quote_count", 0),
                "impressions": metrics.get("impression_count", 0),
                "entities": t.get("entities", {}),
                "context_annotations": t.get("context_annotations", []),
            })

        result = {"tweets": tweets}
        meta = data.get("meta", {})
        if "next_token" in meta:
            result["next_token"] = meta["next_token"]
        result["result_count"] = meta.get("result_count", len(tweets))

        return result

    def search_all_pages(
        self,
        query: str,
        max_total: int = 500,
        since_hours: float = 2
    ) -> list:
        """Paginate through search results up to max_total tweets."""
        all_tweets = []
        next_token = None

        while len(all_tweets) < max_total:
            remaining = max_total - len(all_tweets)
            batch_size = min(remaining, 100)

            result = self.search_recent(
                query=query,
                max_results=batch_size,
                since_hours=since_hours,
                next_token=next_token
            )

            all_tweets.extend(result["tweets"])
            next_token = result.get("next_token")

            if not next_token or result["result_count"] == 0:
                break

            logger.debug(f"Fetched {len(all_tweets)}/{max_total} tweets")

        return all_tweets

    def get_user_tweets(self, user_id: str, max_results: int = 10, since_hours: float = 24) -> list:
        """Get recent tweets from a specific user."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=since_hours)

        params = {
            "max_results": min(max_results, 100),
            "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tweet.fields": "created_at,public_metrics,conversation_id,entities",
        }

        data = self._get(f"users/{user_id}/tweets", params)

        tweets = []
        for t in data.get("data", []):
            metrics = t.get("public_metrics", {})
            tweets.append({
                "tweet_id": t["id"],
                "author_id": user_id,
                "text": t["text"],
                "created_at": t["created_at"],
                "likes": metrics.get("like_count", 0),
                "retweets": metrics.get("retweet_count", 0),
                "replies": metrics.get("reply_count", 0),
                "quotes": metrics.get("quote_count", 0),
                "impressions": metrics.get("impression_count", 0),
            })

        return tweets

    def lookup_users_by_username(self, usernames: list[str]) -> dict:
        """Resolve usernames to user IDs. Returns {username: {id, ...}}."""
        results = {}
        # API allows up to 100 usernames per request
        for i in range(0, len(usernames), 100):
            batch = usernames[i:i + 100]
            params = {
                "usernames": ",".join(batch),
                "user.fields": "id,username,public_metrics,verified,description"
            }
            data = self._get("users/by", params)
            for u in data.get("data", []):
                results[u["username"].lower()] = u

        return results
