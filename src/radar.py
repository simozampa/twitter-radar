"""
Main Twitter Radar engine.
Strategy: scrape tweets from ALL accounts you follow,
let trend detection cluster and surface what's hot.
"""

import os
import time
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from .db import Database
from .twitter_client import TwitterClient
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

        # --- Apify client (primary data source) ---
        self.apify = None
        apify_token = self._resolve_env(self.config.get('apify', {}).get('api_token', ''))
        if apify_token:
            self.apify = ApifyTwitterClient(api_token=apify_token)

        # --- Twitter API client (supplementary) ---
        self.twitter = None
        token = self._resolve_env(self.config.get('twitter', {}).get('bearer_token', ''))
        if token:
            self.twitter = TwitterClient(
                bearer_token=token,
                monthly_credit_limit=self.config.get('twitter', {}).get('monthly_credit_limit', 15000),
            )

        # --- Database ---
        db_path = self.config['storage']['db_path']
        if not os.path.isabs(db_path):
            db_path = str(Path(__file__).parent.parent / db_path)
        self.db = Database(db_path)
        self._data_dir = Path(db_path).parent
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # --- Trend detector ---
        self.detector = TrendDetector(self.config['detection'])

        # --- Following list ---
        self._following_path = self._data_dir / "following.json"
        self._following: list = self._load_following()

        # --- Rotation state ---
        self._rotation_path = self._data_dir / "rotation_state.json"
        self._rotation: dict = self._load_json(self._rotation_path)

    @staticmethod
    def _resolve_env(value: str) -> str:
        if value and value.startswith("${") and value.endswith("}"):
            return os.environ.get(value[2:-1], "")
        return value

    def _load_json(self, path: Path) -> dict:
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_json(self, path: Path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_following(self) -> list:
        if self._following_path.exists():
            try:
                with open(self._following_path) as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _get_scannable_accounts(self) -> list[str]:
        """
        Get all followed accounts eligible for scanning.
        Sorted by follower count (highest first = most likely to have engagement).
        """
        min_followers = self.config.get('following', {}).get('min_followers', 1000)

        accounts = [
            a for a in self._following
            if a.get('followers', 0) >= min_followers
            and a.get('username')
        ]
        accounts.sort(key=lambda a: a.get('followers', 0), reverse=True)

        return [a['username'] for a in accounts]

    def _get_rotation_batch(self, accounts: list[str], batch_size: int) -> list[str]:
        """Get next batch of accounts to scrape, rotating through the full list."""
        if not accounts:
            return []

        idx = self._rotation.get('scan_index', 0)
        start = (idx * batch_size) % len(accounts)
        batch = accounts[start:start + batch_size]

        # Wrap around if needed
        if len(batch) < batch_size:
            batch.extend(accounts[:batch_size - len(batch)])

        # Update rotation
        self._rotation['scan_index'] = idx + 1
        self._save_json(self._rotation_path, self._rotation)

        return batch

    def run_scan(self) -> dict:
        """
        Run a full scan:
        1. Get batch of followed accounts
        2. Scrape their tweets via Apify
        3. Detect trending topics
        """
        if not self.apify:
            logger.error("Apify not configured — can't scan")
            return {}

        all_accounts = self._get_scannable_accounts()
        if not all_accounts:
            logger.error("No following list found. Run refresh_following() first.")
            return {}

        batch_size = self.config.get('following', {}).get('accounts_per_scan', 100)
        batch = self._get_rotation_batch(all_accounts, batch_size)

        logger.info(f"Scanning {len(batch)}/{len(all_accounts)} accounts "
                    f"(rotation {self._rotation.get('scan_index', 0)})")

        # Scrape tweets
        max_tweets = self.config.get('apify', {}).get('max_tweets_per_query', 200)
        tweets = self.apify.scrape_user_tweets(batch, max_tweets=max_tweets)

        # Tag all tweets
        for t in tweets:
            t['community'] = 'following'
            t['is_priority_account'] = True

        # Store
        if tweets:
            self.db.upsert_tweets_batch(tweets)

        logger.info(f"Fetched {len(tweets)} tweets from {len(batch)} accounts")

        # Detect trends across ALL tweets (no category filtering)
        now = time.time()
        trending_hours = self.config['detection']['trending_window_hours']

        # Use stored tweets for baseline (from previous scans)
        baseline_tweets = self.db.get_tweets_in_window(
            'following',
            now - (self.config['detection']['baseline_window_hours'] * 3600),
            now - (trending_hours * 3600)
        )

        trends = self.detector.detect_trends(tweets, baseline_tweets, 'following')

        for trend in trends:
            trend_id = self.db.insert_trend(trend)
            trend['id'] = trend_id

        results = {
            'following': {
                'community_name': 'Your Feed',
                'tweets_fetched': len(tweets),
                'accounts_scanned': len(batch),
                'accounts_total': len(all_accounts),
                'timeline_tweets': 0,
                'apify_tweets': len(tweets),
                'trends': trends,
            }
        }

        self.db.cleanup_old_data(self.config['storage'].get('retention_days', 30))
        return results

    def format_alert(self, results: dict) -> Optional[str]:
        """Format scan results into a Telegram alert."""
        alert_threshold = self.config['alerts'].get('alert_velocity_threshold', 3.0)

        data = results.get('following', {})
        trends = data.get('trends', [])

        if not trends:
            return None

        # Show top trends regardless of velocity on early scans
        hot_trends = [t for t in trends if t['velocity_score'] >= alert_threshold]
        if not hot_trends:
            hot_trends = trends[:5]

        # Filter already-alerted
        new_trends = []
        for t in hot_trends:
            if not self.db.was_alerted_recently(t['topic'], 'following', hours=2):
                new_trends.append(t)

        if not new_trends:
            return None

        lines = [
            "**TREND RADAR**",
            f"_Scanned {data.get('accounts_scanned', 0)}/{data.get('accounts_total', 0)} "
            f"accounts, {data.get('tweets_fetched', 0)} tweets_",
            f"_{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
            "",
        ]

        for i, trend in enumerate(new_trends, 1):
            lines.append(
                f"\n**{i}. {trend['topic']}** "
                f"(velocity: {trend['velocity_score']:.1f}x, "
                f"{trend['tweet_count']} tweets)"
            )

            for tweet in trend.get('top_tweets', [])[:3]:
                url = tweet.get('url', '')
                url_text = f"\n    {url}" if url else ""
                lines.append(
                    f"  → @{tweet.get('author_username', '?')}: "
                    f"{tweet['text'][:150]}..."
                    f"\n    ({tweet.get('likes', 0)} likes, {tweet.get('retweets', 0)} RTs)"
                    f"{url_text}"
                )

            self.db.insert_alert({
                'trend_id': trend.get('id'),
                'community': 'following',
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
