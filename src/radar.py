"""
Twitter Radar v2 — Simplified fetcher.
Handles timeline (X API) + trending (Apify), timestamp filtering, storage.
Trend detection is now done by the LLM, not by clustering.
"""

import os
import time
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional
from requests_oauthlib import OAuth1Session

from .db import Database
from .apify_client import ApifyTwitterClient

logger = logging.getLogger("twitter_radar")


class TwitterRadar:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "config.yaml")

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # --- OAuth 1.0a for timeline ---
        self.oauth = None
        oauth_cfg = self.config.get('twitter_oauth', {})
        api_key = self._resolve_env(oauth_cfg.get('api_key', ''))
        api_secret = self._resolve_env(oauth_cfg.get('api_secret', ''))
        access_token = self._resolve_env(oauth_cfg.get('access_token', ''))
        access_secret = self._resolve_env(oauth_cfg.get('access_token_secret', ''))
        if all([api_key, api_secret, access_token, access_secret]):
            self.oauth = OAuth1Session(api_key, api_secret, access_token, access_secret)
            self.user_id = oauth_cfg.get('user_id', '')

        # --- Apify client ---
        self.apify = None
        apify_token = self._resolve_env(self.config.get('apify', {}).get('api_token', ''))
        if apify_token:
            self.apify = ApifyTwitterClient(api_token=apify_token)

        # --- Database ---
        db_path = self.config['storage']['db_path']
        if not os.path.isabs(db_path):
            db_path = str(Path(__file__).parent.parent / db_path)
        self.db = Database(db_path)
        self._data_dir = Path(db_path).parent
        self._data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_env(value: str) -> str:
        if value and value.startswith("${") and value.endswith("}"):
            return os.environ.get(value[2:-1], "")
        return value

    def _oauth_get(self, url: str, params: dict) -> dict:
        resp = self.oauth.get(url, params=params)
        if resp.status_code == 429:
            reset = int(resp.headers.get('x-rate-limit-reset', time.time() + 60))
            wait = max(reset - time.time() + 1, 1)
            logger.warning(f"Rate limited, waiting {wait:.0f}s")
            time.sleep(wait)
            return self._oauth_get(url, params)
        if resp.status_code != 200:
            logger.error(f"API error {resp.status_code}: {resp.text[:300]}")
            resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # FETCH ALL
    # ------------------------------------------------------------------

    def fetch_all(self, include_trending: bool = True) -> list:
        """Fetch timeline + optionally trending. Returns all tweets."""
        tweets = self.fetch_timeline()
        if include_trending and self.apify:
            tweets += self.fetch_trending()
        return tweets

    # ------------------------------------------------------------------
    # TIMELINE
    # ------------------------------------------------------------------

    def fetch_timeline(self) -> list:
        if not self.oauth:
            logger.error("OAuth not configured")
            return []

        max_results = self.config.get('timeline', {}).get('tweets_per_scan', 200)
        logger.info(f"Fetching timeline ({max_results} tweets)...")

        all_tweets = []
        next_token = None
        remaining = max_results

        while remaining > 0:
            batch = min(remaining, 100)
            params = {
                'max_results': batch,
                'tweet.fields': 'created_at,public_metrics,author_id,entities,conversation_id',
                'user.fields': 'username,public_metrics',
                'expansions': 'author_id',
                'exclude': 'replies',
            }
            if next_token:
                params['pagination_token'] = next_token

            data = self._oauth_get(
                f'https://api.twitter.com/2/users/{self.user_id}/timelines/reverse_chronological',
                params
            )

            users = {}
            for u in data.get('includes', {}).get('users', []):
                users[u['id']] = {
                    'username': u['username'],
                    'followers': u.get('public_metrics', {}).get('followers_count', 0),
                }

            for t in data.get('data', []):
                metrics = t.get('public_metrics', {})
                author = users.get(t['author_id'], {})
                all_tweets.append({
                    'tweet_id': t['id'],
                    'author_id': t['author_id'],
                    'author_username': author.get('username', '?'),
                    'author_followers': author.get('followers', 0),
                    'text': t['text'],
                    'created_at': t.get('created_at', ''),
                    'likes': metrics.get('like_count', 0),
                    'retweets': metrics.get('retweet_count', 0),
                    'replies': metrics.get('reply_count', 0),
                    'quotes': metrics.get('quote_count', 0),
                    'impressions': metrics.get('impression_count', 0),
                    'community': 'timeline',
                    'source': 'x_api',
                })

            meta = data.get('meta', {})
            next_token = meta.get('next_token')
            remaining -= batch
            if not next_token or not data.get('data'):
                break

        logger.info(f"Timeline: {len(all_tweets)} tweets")
        if all_tweets:
            self.db.upsert_tweets_batch(all_tweets)
        return all_tweets

    # ------------------------------------------------------------------
    # TRENDING (Apify)
    # ------------------------------------------------------------------

    def fetch_trending(self) -> list:
        if not self.apify:
            return []

        queries = self.config.get('trending', {}).get('queries', [])
        if not queries:
            return []

        max_tweets = self.config.get('trending', {}).get('max_tweets', 500)
        since_hours = self.config.get('trending', {}).get('since_hours', 12)
        sort_by = self.config.get('trending', {}).get('sort', 'Latest')

        logger.info(f"Fetching trending via Apify (last {since_hours}h, sort={sort_by})...")
        tweets = self.apify.search_tweets(
            queries=queries,
            max_tweets=max_tweets,
            since_hours=since_hours,
            sort_by=sort_by,
        )

        # Post-fetch timestamp filtering
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        pre_filter = len(tweets)
        filtered = []
        for t in tweets:
            t['community'] = 'trending'
            t['source'] = 'apify'
            created = t.get('created_at', '')
            if created:
                try:
                    if 'T' in created:
                        ts = datetime.fromisoformat(created.replace('Z', '+00:00'))
                    else:
                        from email.utils import parsedate_to_datetime
                        ts = parsedate_to_datetime(created)
                    if ts < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass
            filtered.append(t)
        tweets = filtered

        if pre_filter != len(tweets):
            logger.info(f"Timestamp filter: {pre_filter} → {len(tweets)} (removed {pre_filter - len(tweets)} stale)")

        if tweets:
            self.db.upsert_tweets_batch(tweets)
        logger.info(f"Trending: {len(tweets)} tweets via Apify")
        return tweets

    # ------------------------------------------------------------------
    # LAST REPORTED (cross-scan dedup)
    # ------------------------------------------------------------------

    def load_last_reported(self, max_age_hours: float = 24) -> list[dict]:
        path = self._data_dir / "last_reported.json"
        if not path.exists():
            return []
        try:
            with open(path) as f:
                data = json.load(f)
            cutoff = time.time() - (max_age_hours * 3600)
            return [t for t in data.get('reported_topics', []) if t.get('timestamp', 0) > cutoff]
        except Exception:
            return []

    def save_reported_topics(self, topics: list[str]):
        path = self._data_dir / "last_reported.json"
        existing = []
        if path.exists():
            try:
                with open(path) as f:
                    existing = json.load(f).get('reported_topics', [])
            except Exception:
                pass

        now = time.time()
        for topic in topics:
            existing.append({'topic': topic.lower().strip(), 'timestamp': now})
        existing = [t for t in existing if t.get('timestamp', 0) > now - 48 * 3600]

        with open(path, 'w') as f:
            json.dump({
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'reported_topics': existing,
            }, f, indent=2)
