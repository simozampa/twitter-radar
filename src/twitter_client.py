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
    """
    Rate limiter with per-endpoint tracking.
    User timeline endpoint on Basic tier: 10 requests / 15 min.
    User lookup: 300 requests / 15 min.
    """

    def __init__(self):
        self.endpoints: dict[str, dict] = {
            "timeline": {"max": 9, "window": 900, "requests": []},   # 10 limit, stay at 9
            "lookup": {"max": 290, "window": 900, "requests": []},   # 300 limit
            "default": {"max": 280, "window": 900, "requests": []},
        }
        # Track the reset timestamp from API headers
        self._reset_at: float = 0

    def update_from_headers(self, remaining: int, reset_at: int):
        """Update limiter state from API response headers."""
        self._reset_at = reset_at
        if remaining <= 0:
            logger.info(f"API reports 0 remaining, reset at {reset_at}")

    def wait_if_needed(self, endpoint_type: str = "default"):
        ep = self.endpoints.get(endpoint_type, self.endpoints["default"])
        now = time.time()
        ep["requests"] = [t for t in ep["requests"] if now - t < ep["window"]]
        if len(ep["requests"]) >= ep["max"]:
            sleep_time = ep["requests"][0] + ep["window"] - now + 1
            if sleep_time > 0:
                logger.info(f"Rate limit ({endpoint_type}): sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        ep["requests"].append(time.time())

    def can_call(self, endpoint_type: str = "default") -> bool:
        ep = self.endpoints.get(endpoint_type, self.endpoints["default"])
        now = time.time()
        active = [t for t in ep["requests"] if now - t < ep["window"]]
        return len(active) < ep["max"]

    @property
    def timeline_slots_remaining(self) -> int:
        ep = self.endpoints["timeline"]
        now = time.time()
        active = [t for t in ep["requests"] if now - t < ep["window"]]
        return max(0, ep["max"] - len(active))


class TwitterClient:
    def __init__(self, bearer_token: str, monthly_credit_limit: int = 15000):
        self.bearer_token = bearer_token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {bearer_token}",
            "User-Agent": "TwitterRadar/1.0"
        })
        self.limiter = RateLimiter()
        self.credits = CreditTracker(monthly_credit_limit)

    def _get(self, endpoint: str, params: dict, endpoint_type: str = "default") -> dict:
        """Make a rate-limited GET request."""
        self.limiter.wait_if_needed(endpoint_type)
        url = f"{API_BASE}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=30)

        # Update rate limiter from response headers
        remaining = int(resp.headers.get("x-rate-limit-remaining", 999))
        reset = int(resp.headers.get("x-rate-limit-reset", time.time() + 900))
        self.limiter.update_from_headers(remaining, reset)

        if resp.status_code == 429:
            sleep_time = max(reset - time.time() + 1, 1)
            logger.warning(f"429 rate limited ({endpoint_type}). Sleeping {sleep_time:.0f}s")
            time.sleep(sleep_time)
            return self._get(endpoint, params, endpoint_type)

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
            data = self._get("users/by", params, endpoint_type="lookup")
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
            data = self._get(f"users/{user_id}/tweets", params, endpoint_type="timeline")
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
        Fetch timelines for multiple users.
        IMPORTANT: Basic tier only allows 10 timeline requests / 15 min.
        This method fetches as many as the rate limit allows and stops.
        Call again later for remaining accounts.
        
        user_ids: {user_id: username}
        Returns all tweets with author_username attached.
        """
        all_tweets = []
        skipped = []

        for user_id, username in user_ids.items():
            if not self.credits.can_afford(5):
                logger.warning(f"Credits exhausted ({self.credits.remaining} remaining)")
                break

            if not self.limiter.can_call("timeline"):
                skipped.append(username)
                continue

            tweets = self.get_user_timeline(user_id, max_per_user, since_hours)
            for t in tweets:
                t['author_username'] = username
                t['is_priority_account'] = True
            all_tweets.extend(tweets)

            logger.info(f"@{username}: {len(tweets)} tweets "
                       f"(credits: {self.credits.remaining}, "
                       f"timeline slots: {self.limiter.timeline_slots_remaining})")

            # Small delay between calls
            time.sleep(0.5)

        if skipped:
            logger.info(f"Skipped {len(skipped)} accounts (rate limit): "
                       f"{', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''}")

        return all_tweets
