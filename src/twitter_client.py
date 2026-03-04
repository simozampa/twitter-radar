"""
Twitter API v2 client for Trend Radar.
Focused on user timelines (not search) to conserve credits.
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
import requests

logger = logging.getLogger("twitter_radar.client")

API_BASE = "https://api.twitter.com/2"


class CreditTracker:
    """Track API credit usage to avoid blowing the budget."""

    def __init__(self, monthly_limit: int = 15000):
        self.monthly_limit = monthly_limit
        self.used = 0
        self.reset_month = datetime.now(timezone.utc).month

    def _check_reset(self):
        current_month = datetime.now(timezone.utc).month
        if current_month != self.reset_month:
            self.used = 0
            self.reset_month = current_month

    def consume(self, count: int):
        self._check_reset()
        self.used += count
        remaining = self.monthly_limit - self.used
        logger.info(f"Credits: used {count}, total {self.used}/{self.monthly_limit} ({remaining} remaining)")
        if remaining < 1000:
            logger.warning(f"⚠️ Low credits! Only {remaining} remaining this month")

    def can_afford(self, count: int) -> bool:
        self._check_reset()
        return (self.used + count) <= self.monthly_limit

    @property
    def remaining(self) -> int:
        self._check_reset()
        return self.monthly_limit - self.used


class RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, max_requests: int, window_seconds: int = 900):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: list[float] = []

    def wait_if_needed(self):
        now = time.time()
        self.requests = [t for t in self.requests if now - t < self.window_seconds]
        if len(self.requests) >= self.max_requests:
            sleep_time = self.requests[0] + self.window_seconds - now + 1
            if sleep_time > 0:
                logger.info(f"Rate limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        self.requests.append(time.time())


class TwitterClient:
    def __init__(self, bearer_token: str, monthly_credit_limit: int = 15000,
                 rate_limit: int = 280):
        self.bearer_token = bearer_token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {bearer_token}",
            "User-Agent": "TwitterRadar/1.0"
        })
        self.limiter = RateLimiter(rate_limit, 900)
        self.credits = CreditTracker(monthly_credit_limit)

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

    def lookup_users_by_username(self, usernames: list[str]) -> dict:
        """
        Resolve usernames to user IDs.
        Returns {username_lower: {id, username, ...}}.
        Cost: 1 request per 100 usernames (doesn't count against tweet credits).
        """
        results = {}
        for i in range(0, len(usernames), 100):
            batch = usernames[i:i + 100]
            params = {
                "usernames": ",".join(batch),
                "user.fields": "id,username,public_metrics,verified,description"
            }
            data = self._get("users/by", params)
            for u in data.get("data", []):
                results[u["username"].lower()] = u
            errors = data.get("errors", [])
            for e in errors:
                logger.warning(f"User lookup error: {e.get('detail', e)}")

        return results

    def get_user_timeline(self, user_id: str, max_results: int = 10,
                          since_hours: float = 24) -> list:
        """
        Get recent tweets from a specific user's timeline.
        Cost: each tweet returned counts against credits.
        """
        if not self.credits.can_afford(max_results):
            logger.warning(f"Skipping timeline for {user_id}: insufficient credits "
                          f"({self.credits.remaining} remaining)")
            return []

        start_time = datetime.now(timezone.utc) - timedelta(hours=since_hours)

        params = {
            "max_results": max(min(max_results, 100), 5),  # API min is 5
            "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tweet.fields": "created_at,public_metrics,conversation_id,entities",
            "exclude": "replies",  # Only original tweets, not replies
        }

        try:
            data = self._get(f"users/{user_id}/tweets", params)
        except requests.HTTPError as e:
            logger.error(f"Timeline fetch failed for {user_id}: {e}")
            return []

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

        self.credits.consume(len(tweets))
        return tweets

    def get_timelines_batch(self, user_ids: dict, max_per_user: int = 10,
                            since_hours: float = 24) -> list:
        """
        Fetch timelines for multiple users efficiently.
        user_ids: {user_id: username}
        Returns all tweets with author_username attached.
        """
        all_tweets = []

        for user_id, username in user_ids.items():
            if not self.credits.can_afford(5):  # minimum per request
                logger.warning(f"Stopping timeline fetches: credits exhausted "
                              f"({self.credits.remaining} remaining)")
                break

            tweets = self.get_user_timeline(user_id, max_per_user, since_hours)
            for t in tweets:
                t['author_username'] = username
                t['is_priority_account'] = True
            all_tweets.extend(tweets)

            logger.info(f"@{username}: {len(tweets)} tweets "
                       f"(credits remaining: {self.credits.remaining})")

        return all_tweets
