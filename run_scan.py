#!/usr/bin/env python3
"""
CLI entry point for Twitter Radar.
Can be run standalone or called by OpenClaw cron.

Usage:
  python run_scan.py                    # Full scan + print results
  python run_scan.py --json             # Output JSON (for cron integration)
  python run_scan.py --community ai     # Scan specific community only
  python run_scan.py --dry-run          # Fetch + detect but don't alert
"""

import sys
import json
import argparse
import logging
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.radar import TwitterRadar


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )


def main():
    parser = argparse.ArgumentParser(description="Twitter Trend Radar")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--community", default=None, help="Scan specific community")
    parser.add_argument("--dry-run", action="store_true", help="Don't send alerts")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("twitter_radar")

    try:
        radar = TwitterRadar(config_path=args.config)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Run scan
    results = radar.run_scan()

    # Filter to specific community if requested
    if args.community:
        results = {k: v for k, v in results.items() if k == args.community}

    if args.json:
        # JSON output for cron/programmatic use
        output = {}
        for name, data in results.items():
            output[name] = {
                'community_name': data['community_name'],
                'tweets_fetched': data['tweets_fetched'],
                'trends': [
                    {
                        'topic': t['topic'],
                        'velocity_score': t['velocity_score'],
                        'tweet_count': t['tweet_count'],
                        'total_engagement': t['total_engagement'],
                        'keywords': t['keywords'][:10],
                        'top_tweets': [
                            {
                                'author': tw.get('author_username', '?'),
                                'text': tw['text'][:280],
                                'likes': tw.get('likes', 0),
                                'retweets': tw.get('retweets', 0),
                            }
                            for tw in t.get('top_tweets', [])[:3]
                        ]
                    }
                    for t in data.get('trends', [])
                ]
            }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        alert = radar.format_alert(results)
        if alert:
            print(alert)
        else:
            print("No significant trends detected.")

        # Print summary
        for name, data in results.items():
            print(f"\n[{name}] {data['tweets_fetched']} tweets fetched, "
                  f"{len(data.get('trends', []))} trends detected")
            for t in data.get('trends', []):
                print(f"  • {t['topic']} (velocity: {t['velocity_score']:.1f}x, "
                      f"{t['tweet_count']} tweets)")

    # Generate draft prompts for hottest trend (if any)
    if not args.dry_run and not args.json:
        for name, data in results.items():
            trends = data.get('trends', [])
            if trends:
                hottest = trends[0]
                print(f"\n{'='*60}")
                print(f"DRAFT PROMPTS for '{hottest['topic']}':")
                print(f"{'='*60}")
                prompts = radar.get_draft_prompts(hottest)
                for p in prompts[:2]:  # Just show first 2
                    print(f"\n--- {p['style']} ---")
                    print(p['prompt'][:500])
                    print("...")


if __name__ == "__main__":
    main()
