#!/usr/bin/env python3
"""
Twitter Radar v2 — LLM-native trend detection.

Fetches tweets, dedupes RTs, scores by engagement+recency,
outputs the top N tweets for LLM analysis.

Usage:
  python run_scan.py                    # Top tweets, human-readable
  python run_scan.py --json             # JSON output (for cron/LLM)
  python run_scan.py --top 60           # Custom number of top tweets
  python run_scan.py --no-trending      # Skip Apify, timeline only
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.radar import TwitterRadar
from src.trend_detector import dedup_rts, compute_engagement_score


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,  # Keep logs out of stdout for --json
    )


def main():
    parser = argparse.ArgumentParser(description="Twitter Trend Radar v2")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--json", action="store_true", help="Output JSON for LLM")
    parser.add_argument("--top", type=int, default=60, help="Number of top tweets to output")
    parser.add_argument("--no-trending", action="store_true", help="Skip Apify trending")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("twitter_radar")

    try:
        radar = TwitterRadar(config_path=args.config)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Fetch tweets
    all_tweets = radar.fetch_all(include_trending=not args.no_trending)

    # Dedup RTs
    deduped = dedup_rts(all_tweets)
    logger.info(f"After RT dedup: {len(all_tweets)} → {len(deduped)} tweets")

    # Score and sort by engagement (recency-weighted)
    for t in deduped:
        t['engagement_score'] = compute_engagement_score(t)
    deduped.sort(key=lambda t: t['engagement_score'], reverse=True)

    # Take top N
    top = deduped[:args.top]

    if args.json:
        output = {
            'scan_stats': {
                'total_fetched': len(all_tweets),
                'after_dedup': len(deduped),
                'top_returned': len(top),
                'timeline_count': sum(1 for t in all_tweets if t.get('source') != 'apify'),
                'apify_count': sum(1 for t in all_tweets if t.get('source') == 'apify'),
            },
            'tweets': [
                {
                    'author': t.get('display_author', t.get('author_username', '?')),
                    'text': t.get('original_text', t.get('text', ''))[:500],
                    'likes': t.get('likes', 0),
                    'retweets': t.get('retweets', 0),
                    'replies': t.get('replies', 0),
                    'quotes': t.get('quotes', 0),
                    'created_at': t.get('created_at', ''),
                    'source': t.get('source', ''),
                    'is_rt': t.get('is_rt', False),
                    'engagement_score': round(t.get('engagement_score', 0), 1),
                }
                for t in top
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Twitter Radar — {len(all_tweets)} tweets fetched, "
              f"{len(deduped)} after dedup, showing top {len(top)}\n")
        for i, t in enumerate(top, 1):
            author = t.get('display_author', t.get('author_username', '?'))
            text = t.get('original_text', t.get('text', ''))[:200]
            likes = t.get('likes', 0)
            rts = t.get('retweets', 0)
            score = t.get('engagement_score', 0)
            rt_flag = " [RT]" if t.get('is_rt') else ""
            print(f"{i:2d}. @{author}{rt_flag} (♥{likes} ↻{rts} score:{score:.0f})")
            print(f"    {text}\n")


if __name__ == "__main__":
    main()
