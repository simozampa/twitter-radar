"""
Main Twitter Radar engine.
Two-pronged strategy:
  1. Twitter API: Monitor specific high-value account timelines (precise, credit-conscious)
  2. Apify: Broad search + trend discovery (no Twitter credit cost)
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
from .draft_generator import build_trend_context, generate_draft_prompt, format_drafts_for_alert

logger = logging.getLogger("twitter_radar")


class TwitterRadar:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "config.yaml")

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # --- Twitter API client (for timelines) ---
        self.twitter = None
        token = self._resolve_env(self.config['twitter'].get('bearer_token', ''))
        if token:
            self.twitter = TwitterClient(
                bearer_token=token,
                monthly_credit_limit=self.config['twitter'].get('monthly_credit_limit', 15000),
                rate_limit=self.config['twitter'].get('search_rate_limit', 280)
            )

        # --- Apify client (for broad discovery) ---
        self.apify = None
        apify_token = self._resolve_env(self.config.get('apify', {}).get('api_token', ''))
        if apify_token:
            self.apify = ApifyTwitterClient(api_token=apify_token)

        # --- Database ---
        db_path = self.config['storage']['db_path']
        if not os.path.isabs(db_path):
            db_path = str(Path(__file__).parent.parent / db_path)
        self.db = Database(db_path)

        # --- Trend detector ---
        self.detector = TrendDetector(self.config['detection'])

        # --- Resolved user IDs cache ---
        self._user_id_cache: dict = {}

    @staticmethod
    def _resolve_env(value: str) -> str:
        """Resolve ${ENV_VAR} references."""
        if value and value.startswith("${") and value.endswith("}"):
            return os.environ.get(value[2:-1], "")
        return value

    def get_enabled_communities(self) -> dict:
        return {
            name: cfg for name, cfg in self.config['communities'].items()
            if cfg.get('enabled', True)
        }

    def _resolve_user_ids(self, usernames: list[str]) -> dict:
        """
        Resolve usernames to Twitter user IDs.
        Uses cache to avoid repeated lookups.
        Returns {user_id: username}.
        """
        if not self.twitter:
            return {}

        # Filter out already-cached
        to_resolve = [u for u in usernames if u.lower() not in self._user_id_cache]

        if to_resolve:
            resolved = self.twitter.lookup_users_by_username(to_resolve)
            for username_lower, user_data in resolved.items():
                self._user_id_cache[username_lower] = user_data['id']
                logger.info(f"Resolved @{username_lower} → {user_data['id']}")

        # Build {id: username} mapping
        result = {}
        for u in usernames:
            uid = self._user_id_cache.get(u.lower())
            if uid:
                result[uid] = u
            else:
                logger.warning(f"Could not resolve @{u}")

        return result

    def fetch_timelines(self, community_name: str, community_config: dict) -> list:
        """
        Fetch tweets from priority accounts via Twitter API timelines.
        Very credit-efficient: only reads from accounts we care about.
        """
        if not self.twitter:
            logger.warning("Twitter API not configured, skipping timelines")
            return []

        accounts = community_config.get('priority_accounts', [])
        if not accounts:
            return []

        # Resolve usernames to IDs
        user_ids = self._resolve_user_ids(accounts)
        if not user_ids:
            return []

        max_per_user = self.config['twitter'].get('max_tweets_per_user', 10)
        since_hours = self.config['detection']['baseline_window_hours']

        tweets = self.twitter.get_timelines_batch(user_ids, max_per_user, since_hours)

        # Tag with community
        for t in tweets:
            t['community'] = community_name
            t['is_priority_account'] = True

        if tweets:
            self.db.upsert_tweets_batch(tweets)

        logger.info(f"[{community_name}] Timelines: {len(tweets)} tweets from "
                    f"{len(user_ids)} accounts (credits remaining: "
                    f"{self.twitter.credits.remaining})")

        return tweets

    def fetch_apify(self, community_name: str, community_config: dict) -> list:
        """
        Fetch tweets via Apify for broad discovery.
        No Twitter API credits consumed.
        """
        if not self.apify:
            logger.warning("Apify not configured, skipping broad search")
            return []

        queries = community_config.get('queries', [])
        if not queries:
            return []

        max_tweets = self.config.get('apify', {}).get('max_tweets_per_query', 100)

        # Search for top tweets (engagement-sorted)
        tweets = self.apify.search_tweets(
            queries=queries,
            max_tweets=max_tweets * len(queries),
            since_hours=self.config['detection']['baseline_window_hours'],
            sort_by="Top"
        )

        # Tag with community and priority status
        priority_usernames = set(
            u.lower() for u in community_config.get('priority_accounts', [])
        )
        for t in tweets:
            t['community'] = community_name
            t['is_priority_account'] = t.get('author_username', '').lower() in priority_usernames

        if tweets:
            self.db.upsert_tweets_batch(tweets)

        logger.info(f"[{community_name}] Apify: {len(tweets)} tweets from broad search")
        return tweets

    def detect_trends_for_community(self, name: str, all_tweets: list) -> list:
        """Run trend detection on fetched tweets."""
        now = time.time()
        trending_hours = self.config['detection']['trending_window_hours']

        # Split into current vs baseline based on created_at or fetched_at
        current_tweets = []
        baseline_tweets = []
        cutoff = now - (trending_hours * 3600)

        for t in all_tweets:
            if t.get('fetched_at', now) >= cutoff:
                current_tweets.append(t)
            else:
                baseline_tweets.append(t)

        # If we can't split well (all tweets are "current"), use engagement as signal
        if not baseline_tweets and current_tweets:
            # Use all tweets as current, empty baseline = everything looks trending
            # Better: split by engagement percentile
            logger.info(f"[{name}] No baseline window, using all {len(current_tweets)} tweets as current")

        logger.info(f"[{name}] Detecting trends: {len(current_tweets)} current, "
                    f"{len(baseline_tweets)} baseline")

        trends = self.detector.detect_trends(current_tweets, baseline_tweets, name)

        for trend in trends:
            trend_id = self.db.insert_trend(trend)
            trend['id'] = trend_id

        return trends

    def run_scan(self, use_twitter: bool = True, use_apify: bool = True) -> dict:
        """
        Run a full scan using configured data sources.
        """
        results = {}
        communities = self.get_enabled_communities()

        for name, cfg in communities.items():
            logger.info(f"{'='*40}")
            logger.info(f"Scanning community: {cfg.get('name', name)}")

            all_tweets = []

            # 1. Twitter API: precise timeline monitoring
            if use_twitter and self.twitter:
                timeline_tweets = self.fetch_timelines(name, cfg)
                all_tweets.extend(timeline_tweets)

            # 2. Apify: broad discovery
            if use_apify and self.apify:
                apify_tweets = self.fetch_apify(name, cfg)
                all_tweets.extend(apify_tweets)

            # 3. Detect trends
            trends = self.detect_trends_for_community(name, all_tweets)

            results[name] = {
                'community_name': cfg.get('name', name),
                'tweets_fetched': len(all_tweets),
                'timeline_tweets': len([t for t in all_tweets if t.get('source') != 'apify']),
                'apify_tweets': len([t for t in all_tweets if t.get('source') == 'apify']),
                'trends': trends,
            }

        self.db.cleanup_old_data(self.config['storage'].get('retention_days', 30))
        return results

    def format_alert(self, results: dict) -> Optional[str]:
        """Format scan results into a Telegram alert message."""
        alert_threshold = self.config['alerts'].get('alert_velocity_threshold', 5.0)

        sections = []
        has_trends = False

        for name, data in results.items():
            trends = data.get('trends', [])
            # For initial runs, show all trends (velocity threshold is less useful
            # without a proper baseline)
            hot_trends = [t for t in trends if t['velocity_score'] >= alert_threshold]

            # If no hot trends, show top trends anyway if they exist
            if not hot_trends and trends:
                hot_trends = trends[:3]

            if not hot_trends:
                continue

            new_trends = []
            for t in hot_trends:
                if not self.db.was_alerted_recently(t['topic'], name, hours=2):
                    new_trends.append(t)

            if not new_trends:
                continue

            has_trends = True
            community_name = data.get('community_name', name)
            section = [f"\n**{community_name}** ({len(new_trends)} trending)"]
            section.append(f"_Tweets scanned: {data['tweets_fetched']} "
                         f"(API: {data['timeline_tweets']}, Apify: {data['apify_tweets']})_")

            for i, trend in enumerate(new_trends, 1):
                section.append(
                    f"\n{i}. **{trend['topic']}** "
                    f"(velocity: {trend['velocity_score']:.1f}x, "
                    f"{trend['tweet_count']} tweets)"
                )

                for tweet in trend.get('top_tweets', [])[:3]:
                    url = tweet.get('url', '')
                    url_text = f" [{url}]" if url else ""
                    section.append(
                        f"  → @{tweet.get('author_username', '?')}: "
                        f"{tweet['text'][:150]}... "
                        f"({tweet.get('likes', 0)}❤️ {tweet.get('retweets', 0)}🔁)"
                        f"{url_text}"
                    )

                self.db.insert_alert({
                    'trend_id': trend.get('id'),
                    'community': name,
                    'topic': trend['topic'],
                    'velocity_score': trend['velocity_score'],
                    'drafts': [],
                })

            sections.append("\n".join(section))

        if not has_trends:
            return None

        header = "**TWITTER TREND RADAR**\n"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        header += f"_Scan: {timestamp}_"

        if self.twitter:
            header += f"\n_Twitter API credits remaining: {self.twitter.credits.remaining}_"

        return header + "\n" + "\n---\n".join(sections)

    def get_draft_prompts(self, trend: dict) -> list[dict]:
        """Get draft generation prompts for a trend."""
        styles = self.config.get('drafts', {}).get('styles', [])
        prompts = []
        for style in styles:
            prompt = generate_draft_prompt(trend, style)
            prompts.append({
                'style': style['name'],
                'prompt': prompt,
                'max_length': style.get('max_length', 1000),
            })
        return prompts

    def is_quiet_hours(self) -> bool:
        quiet = self.config['alerts'].get('quiet_hours', {})
        if not quiet:
            return False
        now_hour = datetime.now().hour
        start = quiet.get('start', 23)
        end = quiet.get('end', 8)
        if start > end:
            return now_hour >= start or now_hour < end
        else:
            return start <= now_hour < end
