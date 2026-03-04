"""
Main Twitter Radar engine.
Orchestrates fetching, trend detection, and alert formatting.
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
from .trend_detector import TrendDetector, compute_engagement_score
from .draft_generator import build_trend_context, generate_draft_prompt, format_drafts_for_alert

logger = logging.getLogger("twitter_radar")


class TwitterRadar:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "config.yaml")

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Resolve bearer token from env if needed
        token = self.config['twitter']['bearer_token']
        if token.startswith("${") and token.endswith("}"):
            env_var = token[2:-1]
            token = os.environ.get(env_var, "")
            if not token:
                raise ValueError(f"Environment variable {env_var} not set")

        self.client = TwitterClient(
            bearer_token=token,
            rate_limit=self.config['twitter'].get('search_rate_limit', 280)
        )

        db_path = self.config['storage']['db_path']
        if not os.path.isabs(db_path):
            db_path = str(Path(__file__).parent.parent / db_path)
        self.db = Database(db_path)

        self.detector = TrendDetector(self.config['detection'])

    def get_enabled_communities(self) -> dict:
        """Get all enabled community configs."""
        return {
            name: cfg for name, cfg in self.config['communities'].items()
            if cfg.get('enabled', True)
        }

    def fetch_community(self, name: str, community_config: dict) -> list:
        """Fetch tweets for a community and store them."""
        all_tweets = []
        priority_usernames = set(
            u.lower() for u in community_config.get('priority_accounts', [])
        )
        min_likes = community_config.get('min_likes', 0)
        min_rts = community_config.get('min_retweets', 0)

        for query in community_config.get('queries', []):
            # Add minimum engagement filter to query
            full_query = f"{query} min_faves:{min_likes}"

            try:
                tweets = self.client.search_all_pages(
                    query=full_query,
                    max_total=200,
                    since_hours=self.config['detection']['baseline_window_hours']
                )

                for tweet in tweets:
                    tweet['community'] = name
                    tweet['is_priority_account'] = (
                        tweet.get('author_username', '').lower() in priority_usernames
                    )

                all_tweets.extend(tweets)
                self.db.log_fetch(name, query, len(tweets))
                logger.info(f"[{name}] Query '{query[:50]}...' → {len(tweets)} tweets")

            except Exception as e:
                logger.error(f"[{name}] Error fetching query '{query[:50]}...': {e}")
                continue

        # Store tweets
        if all_tweets:
            self.db.upsert_tweets_batch(all_tweets)
            logger.info(f"[{name}] Total: {len(all_tweets)} tweets stored")

        return all_tweets

    def detect_trends_for_community(self, name: str, community_config: dict) -> list:
        """Run trend detection for a community."""
        now = time.time()
        trending_hours = self.config['detection']['trending_window_hours']
        baseline_hours = self.config['detection']['baseline_window_hours']

        current_tweets = self.db.get_tweets_in_window(
            name,
            now - (trending_hours * 3600),
            now
        )
        baseline_tweets = self.db.get_tweets_in_window(
            name,
            now - (baseline_hours * 3600),
            now - (trending_hours * 3600)
        )

        logger.info(
            f"[{name}] Detecting trends: {len(current_tweets)} current, "
            f"{len(baseline_tweets)} baseline tweets"
        )

        trends = self.detector.detect_trends(current_tweets, baseline_tweets, name)

        # Store trends
        for trend in trends:
            trend_id = self.db.insert_trend(trend)
            trend['id'] = trend_id

        return trends

    def run_scan(self) -> dict:
        """
        Run a full scan: fetch tweets for all enabled communities,
        detect trends, and return results.
        """
        results = {}
        communities = self.get_enabled_communities()

        for name, cfg in communities.items():
            logger.info(f"Scanning community: {name}")

            # Fetch fresh data
            tweets = self.fetch_community(name, cfg)

            # Detect trends
            trends = self.detect_trends_for_community(name, cfg)

            results[name] = {
                'community_name': cfg.get('name', name),
                'tweets_fetched': len(tweets),
                'trends': trends,
            }

        # Cleanup old data periodically
        self.db.cleanup_old_data(
            self.config['storage'].get('retention_days', 30)
        )

        return results

    def format_alert(self, results: dict) -> Optional[str]:
        """
        Format scan results into a Telegram alert message.
        Returns None if nothing worth alerting.
        """
        alert_threshold = self.config['alerts'].get('alert_velocity_threshold', 5.0)
        include_drafts = self.config['alerts'].get('include_drafts', True)

        sections = []
        has_trends = False

        for name, data in results.items():
            trends = data.get('trends', [])
            hot_trends = [t for t in trends if t['velocity_score'] >= alert_threshold]

            if not hot_trends:
                continue

            # Filter out already-alerted topics
            new_trends = []
            for t in hot_trends:
                if not self.db.was_alerted_recently(t['topic'], name, hours=2):
                    new_trends.append(t)

            if not new_trends:
                continue

            has_trends = True
            community_name = data.get('community_name', name)
            section = [f"\n**{community_name}** ({len(new_trends)} trending)"]

            for i, trend in enumerate(new_trends, 1):
                section.append(
                    f"\n{i}. **{trend['topic']}** "
                    f"(velocity: {trend['velocity_score']:.1f}x, "
                    f"{trend['tweet_count']} tweets)"
                )

                # Add top tweets
                for tweet in trend.get('top_tweets', [])[:3]:
                    engagement = (
                        tweet.get('likes', 0) + tweet.get('retweets', 0) * 2
                    )
                    section.append(
                        f"  → @{tweet.get('author_username', '?')}: "
                        f"{tweet['text'][:150]}... "
                        f"({tweet.get('likes', 0)}❤️ {tweet.get('retweets', 0)}🔁)"
                    )

                # Include draft generation prompts (to be filled by the caller)
                if include_drafts:
                    section.append(f"\n  [Draft prompts available for this trend]")

                # Log the alert
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

        header = "**🔥 TWITTER TREND RADAR**\n"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        header += f"*Scan: {timestamp}*"

        return header + "\n" + "\n---\n".join(sections)

    def get_draft_prompts(self, trend: dict) -> list[dict]:
        """
        Get draft generation prompts for a trend.
        Returns list of {style, prompt} dicts to be sent to an LLM.
        """
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
        """Check if we're in quiet hours."""
        quiet = self.config['alerts'].get('quiet_hours', {})
        if not quiet:
            return False

        tz_name = quiet.get('timezone', 'UTC')
        # Simple hour check — we use the system clock
        now_hour = datetime.now().hour  # Local time
        start = quiet.get('start', 23)
        end = quiet.get('end', 8)

        if start > end:  # Wraps midnight
            return now_hour >= start or now_hour < end
        else:
            return start <= now_hour < end
