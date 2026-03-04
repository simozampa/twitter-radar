"""
Twitter Trend Radar — Final Architecture
=========================================
1. X API: Pull your home timeline (200 tweets every 4 hours)
2. Apify: Broad trending topics scan (1-2x daily for discovery)
3. Trend detection: Surface what's hot, generate draft takes
"""

import os
import time
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from requests_oauthlib import OAuth1Session

from .db import Database
from .apify_client import ApifyTwitterClient
from .trend_detector import TrendDetector, compute_engagement_score
from .draft_generator import build_trend_context, generate_draft_prompt

logger = logging.getLogger("twitter_radar")


class TwitterRadar:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "config.yaml")

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # --- OAuth 1.0a for timeline (user-context auth as zampa0x) ---
        self.oauth = None
        oauth_cfg = self.config.get('twitter_oauth', {})
        api_key = self._resolve_env(oauth_cfg.get('api_key', ''))
        api_secret = self._resolve_env(oauth_cfg.get('api_secret', ''))
        access_token = self._resolve_env(oauth_cfg.get('access_token', ''))
        access_secret = self._resolve_env(oauth_cfg.get('access_token_secret', ''))
        if all([api_key, api_secret, access_token, access_secret]):
            self.oauth = OAuth1Session(api_key, api_secret, access_token, access_secret)
            self.user_id = oauth_cfg.get('user_id', '')

        # --- Bearer token for following list refresh ---
        self.bearer_token = self._resolve_env(
            self.config.get('twitter', {}).get('bearer_token', ''))

        # --- Apify client (trending discovery) ---
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

        # --- Trend detector ---
        self.detector = TrendDetector(self.config['detection'])

        # --- Following list (cached) ---
        self._following_path = self._data_dir / "following.json"

    @staticmethod
    def _resolve_env(value: str) -> str:
        if value and value.startswith("${") and value.endswith("}"):
            return os.environ.get(value[2:-1], "")
        return value

    def _oauth_get(self, url: str, params: dict) -> dict:
        """Make an authenticated GET request via OAuth 1.0a."""
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
    # TIMELINE (X API — your curated feed)
    # ------------------------------------------------------------------

    def fetch_timeline(self, max_results: int = 200) -> list:
        """
        Pull your home timeline via X API.
        Cost: $0.005 per tweet.
        """
        if not self.oauth:
            logger.error("OAuth not configured — can't fetch timeline")
            return []

        logger.info(f"Fetching timeline ({max_results} tweets)...")

        all_tweets = []
        next_token = None
        remaining = max_results

        while remaining > 0:
            batch = min(remaining, 100)  # API max per request
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

            # Build user lookup
            users = {}
            for u in data.get('includes', {}).get('users', []):
                users[u['id']] = {
                    'username': u['username'],
                    'followers': u.get('public_metrics', {}).get('followers_count', 0),
                }

            # Parse tweets
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
                    'is_priority_account': True,
                    'source': 'x_api',
                })

            meta = data.get('meta', {})
            next_token = meta.get('next_token')
            remaining -= batch

            if not next_token or not data.get('data'):
                break

        logger.info(f"Timeline: {len(all_tweets)} tweets fetched "
                    f"(est. cost: ${len(all_tweets) * 0.005:.2f})")

        # Store
        if all_tweets:
            self.db.upsert_tweets_batch(all_tweets)

        return all_tweets

    # ------------------------------------------------------------------
    # TRENDING (Apify — broad discovery)
    # ------------------------------------------------------------------

    def fetch_trending(self) -> list:
        """
        Fetch broad trending topics via Apify.
        Catches things outside your bubble.
        Cost: ~$0.08 per run.
        """
        if not self.apify:
            logger.warning("Apify not configured, skipping trending")
            return []

        topics = self.config.get('trending', {}).get('queries', [])
        if not topics:
            return []

        max_tweets = self.config.get('trending', {}).get('max_tweets', 100)

        logger.info(f"Fetching trending topics via Apify...")
        tweets = self.apify.search_tweets(
            queries=topics,
            max_tweets=max_tweets,
            sort_by="Top"
        )

        for t in tweets:
            t['community'] = 'trending'
            t['is_priority_account'] = False
            t['source'] = 'apify'

        if tweets:
            self.db.upsert_tweets_batch(tweets)

        logger.info(f"Trending: {len(tweets)} tweets via Apify")
        return tweets

    # ------------------------------------------------------------------
    # FOLLOWING LIST REFRESH (X API bearer token)
    # ------------------------------------------------------------------

    def refresh_following(self) -> int:
        """
        Refresh the following list via X API.
        Cost: $0.01 per user (~$14.54 for 1,454 users).
        Do this monthly.
        """
        import requests

        if not self.bearer_token:
            logger.error("No bearer token for following refresh")
            return 0

        headers = {'Authorization': f'Bearer {self.bearer_token}'}
        all_users = []
        next_token = None

        while True:
            params = {
                'max_results': 1000,
                'user.fields': 'id,username,public_metrics,description',
            }
            if next_token:
                params['pagination_token'] = next_token

            resp = requests.get(
                f'https://api.twitter.com/2/users/{self.user_id}/following',
                headers=headers, params=params, timeout=30
            )

            if resp.status_code == 429:
                reset = int(resp.headers.get('x-rate-limit-reset', time.time() + 60))
                time.sleep(max(reset - time.time() + 1, 1))
                continue

            if resp.status_code != 200:
                logger.error(f"Following refresh failed: {resp.status_code}")
                break

            data = resp.json()
            for u in data.get('data', []):
                pm = u.get('public_metrics', {})
                all_users.append({
                    'user_id': u['id'],
                    'username': u['username'],
                    'name': u.get('name', ''),
                    'followers': pm.get('followers_count', 0),
                    'description': u.get('description', '')[:200],
                })

            next_token = data.get('meta', {}).get('next_token')
            if not next_token:
                break

        if all_users:
            with open(self._following_path, 'w') as f:
                json.dump(all_users, f, indent=2)
            logger.info(f"Following list refreshed: {len(all_users)} accounts")

        return len(all_users)

    # ------------------------------------------------------------------
    # SCAN + TREND DETECTION
    # ------------------------------------------------------------------

    def run_scan(self, include_trending: bool = False) -> dict:
        """
        Run a scan:
        1. Pull timeline (always)
        2. Optionally fetch trending via Apify
        3. Detect trends
        """
        timeline_cfg = self.config.get('timeline', {})
        max_tweets = timeline_cfg.get('tweets_per_scan', 200)

        # 1. Fetch timeline
        timeline_tweets = self.fetch_timeline(max_results=max_tweets)

        # 2. Optionally fetch trending
        trending_tweets = []
        if include_trending and self.apify:
            trending_tweets = self.fetch_trending()

        all_tweets = timeline_tweets + trending_tweets

        # 3. Detect trends
        now = time.time()
        trending_hours = self.config['detection']['trending_window_hours']
        baseline_hours = self.config['detection']['baseline_window_hours']

        # Use stored tweets for baseline
        baseline_tweets = self.db.get_tweets_in_window(
            'timeline',
            now - (baseline_hours * 3600),
            now - (trending_hours * 3600)
        )

        trends = self.detector.detect_trends(all_tweets, baseline_tweets, 'timeline')

        for trend in trends:
            trend_id = self.db.insert_trend(trend)
            trend['id'] = trend_id

        results = {
            'feed': {
                'community_name': 'Your Feed',
                'tweets_fetched': len(all_tweets),
                'timeline_tweets': len(timeline_tweets),
                'trending_tweets': len(trending_tweets),
                'apify_tweets': len(trending_tweets),
                'trends': trends,
                'est_cost': f"${len(timeline_tweets) * 0.005:.2f}",
            }
        }

        self.db.cleanup_old_data(self.config['storage'].get('retention_days', 30))
        return results

    # ------------------------------------------------------------------
    # ALERT FORMATTING
    # ------------------------------------------------------------------

    def format_alert(self, results: dict) -> Optional[str]:
        """Format scan results into a Telegram alert."""
        data = results.get('feed', {})
        trends = data.get('trends', [])

        if not trends:
            return None

        # Show top trends
        alert_threshold = self.config['alerts'].get('alert_velocity_threshold', 3.0)
        hot_trends = [t for t in trends if t['velocity_score'] >= alert_threshold]
        if not hot_trends:
            hot_trends = trends[:5]

        # Filter already-alerted
        new_trends = []
        for t in hot_trends:
            if not self.db.was_alerted_recently(t['topic'], 'timeline', hours=2):
                new_trends.append(t)

        if not new_trends:
            return None

        trending_count = data.get('trending_tweets', 0)
        trending_str = f" + {trending_count} trending" if trending_count else ""
        lines = [
            "**TREND RADAR**",
            f"_{data.get('timeline_tweets', 0)} timeline tweets{trending_str}"
            f" | cost: {data.get('est_cost', '?')}_",
            "",
        ]

        for i, trend in enumerate(new_trends, 1):
            lines.append(
                f"**{i}. {trend['topic']}** "
                f"({trend['tweet_count']} tweets, "
                f"velocity: {trend['velocity_score']:.1f}x)"
            )

            for tweet in trend.get('top_tweets', [])[:3]:
                lines.append(
                    f"  → @{tweet.get('author_username', '?')}: "
                    f"{tweet['text'][:150]}"
                    f"\n    {tweet.get('likes', 0)} likes, {tweet.get('retweets', 0)} RTs"
                )
            lines.append("")

            self.db.insert_alert({
                'trend_id': trend.get('id'),
                'community': 'timeline',
                'topic': trend['topic'],
                'velocity_score': trend['velocity_score'],
                'drafts': [],
            })

        return "\n".join(lines)

    def get_draft_prompts(self, trend: dict) -> list[dict]:
        """Get draft generation prompts for a trend."""
        styles = self.config.get('drafts', {}).get('styles', [])
        return [
            {
                'style': s['name'],
                'prompt': generate_draft_prompt(trend, s),
                'max_length': s.get('max_length', 1000),
            }
            for s in styles
        ]

    def is_quiet_hours(self) -> bool:
        quiet = self.config['alerts'].get('quiet_hours', {})
        if not quiet:
            return False
        now_hour = datetime.now().hour
        start = quiet.get('start', 23)
        end = quiet.get('end', 8)
        if start > end:
            return now_hour >= start or now_hour < end
        return start <= now_hour < end
